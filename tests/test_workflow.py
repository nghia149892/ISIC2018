import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ISIC2018.src.benchmarking.metrics import classification_metrics
from ISIC2018.src.benchmarking.runner import benchmark_batch
from ISIC2018.src.checkpoints import find_checkpoint
from ISIC2018.src.config import cfg
from ISIC2018.src.data.dataset import create_dataloaders
from ISIC2018.src.data.labels import load_ground_truth, read_labels
from ISIC2018.src.data.transforms import get_transforms, preprocess
from ISIC2018.src.inference.backends import InferenceBackend
from ISIC2018.src.models.classifiers import StudentModel
from ISIC2018.src.training.loops import fit, fit_kd
from ISIC2018.src.training.optimization import EarlyStopping, create_optimizer
from ISIC2018.src.training.experiment import main as train_experiment
from ISIC2018.src.benchmarking.cli import parse_args, select_engines
from ISIC2018.__main__ import main as project_cli


class WorkflowTests(unittest.TestCase):
    def test_project_cli_routes_backend_arguments_and_training_kind(self):
        with patch("ISIC2018.__main__.import_module") as load:
            project_cli(["benchmark", "--cpu-only", "--limit", "5"])
            load.assert_called_once_with(".src.benchmarking.cli", package="ISIC2018")
            load.return_value.main.assert_called_once_with(["--cpu-only", "--limit", "5"])
        with patch("ISIC2018.__main__.import_module") as load:
            project_cli(["train", "student-kd", "--epochs", "1"])
            load.return_value.main.assert_called_once_with("student_kd", ["--epochs", "1"])

    def test_cpu_benchmark_does_not_require_tensorrt(self):
        args = parse_args(["--cpu-only"])
        with patch("ISIC2018.src.benchmarking.cli.torch.cuda.is_available", return_value=False):
            self.assertEqual(select_engines(args), {})
            self.assertEqual(select_engines(parse_args([])), {})

    def test_training_entry_point_records_manifest(self):
        class TinyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 8)
                self.fc = nn.Linear(8, 7)

            def forward(self, images):
                return self.fc(self.backbone(images.mean(dim=(2, 3))))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "truth.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image", *cfg.CLASS_LABELS])
                for index in range(7):
                    for count in range(5):
                        name = f"ISIC_{index}_{count}"
                        Image.new("RGB", (24, 24), (index * 30, count * 20, 80)).save(root / f"{name}.jpg")
                        writer.writerow([name, *np.eye(7, dtype=int)[index]])
            output = root / "student.pth"
            with patch.object(cfg, "TRAIN_GROUND_TRUTH", csv_path), patch.object(cfg, "TRAIN_IMAGE_DIR", root):
                with patch("ISIC2018.src.training.experiment.StudentModel", return_value=TinyStudent()):
                    train_experiment("student", ["--epochs", "1", "--workers", "0", "--batch-size", "4", "--no-pretrained", "--output", str(output)])
            self.assertTrue(output.is_file())
            manifest = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(len(manifest["train_images"]), 28)
            self.assertEqual(len(manifest["validation_images"]), 7)
            self.assertFalse(set(manifest["train_images"]) & set(manifest["validation_images"]))
            self.assertEqual(manifest["class_labels"], list(cfg.CLASS_LABELS))

    def test_metrics_match_sklearn_with_missing_class(self):
        targets = np.array([0, 0, 1, 2, 3, 4, 5])
        predictions = np.array([0, 1, 1, 2, 3, 5, 5])
        result = classification_metrics(np.eye(7)[predictions], targets)
        self.assertAlmostEqual(result["accuracy"], accuracy_score(targets, predictions))
        self.assertAlmostEqual(result["macro_f1"], f1_score(targets, predictions, labels=range(7), average="macro", zero_division=0))
        self.assertIsNone(result["per_class_recall"]["VASC"]["recall"])

    def test_ground_truth_order_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truth.csv"
            rows = [["ISIC_A", *np.eye(7, dtype=int)[0]], ["ISIC_B", *np.eye(7, dtype=int)[5]]]
            def write(values):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["image", *cfg.CLASS_LABELS])
                    writer.writerows(values)
            write(rows)
            actual = load_ground_truth(path, [Path("ISIC_B.jpg"), Path("ISIC_A.jpg")])
            np.testing.assert_array_equal(actual, [5, 0])
            write([rows[0], rows[0]])
            with self.assertRaises(ValueError):
                read_labels(path)
            write([["ISIC_A", 1, 1, 0, 0, 0, 0, 0]])
            with self.assertRaises(ValueError):
                read_labels(path)

    def test_checkpoint_precedence_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train").mkdir()
            trained = root / "train/best_student_model_withKD.pth"
            legacy = root / "best_model_withKD.pth"
            trained.touch()
            legacy.touch()
            with patch.object(cfg, "WEIGHTS_DIR", root):
                with self.assertWarns(UserWarning):
                    self.assertEqual(find_checkpoint(), trained)
                self.assertEqual(find_checkpoint(legacy), legacy)

    def test_preprocess_matches_validation_and_partial_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.png"
            image = Image.new("RGB", (41, 37), (10, 70, 140))
            image.save(path)
            _, transform = get_transforms()
            np.testing.assert_array_equal(preprocess(path)[0], transform(image).numpy())
            seen = []
            def predict(array):
                seen.append(len(array))
                return np.tile(np.eye(7)[5], (len(array), 1))
            backend = InferenceBackend("PyTorch CPU", predict, {})
            result = benchmark_batch([backend], [path] * 5, 4, warmup=0, targets=np.full(5, 5))
            self.assertEqual(seen, [4, 1])
            self.assertEqual(result["backends"]["PyTorch CPU"]["latency"]["timed_images"], 5)
            self.assertEqual(result["backends"]["PyTorch CPU"]["classification"]["accuracy"], 1)

    def test_training_split_and_optimizer_match_between_students(self):
        frame = pd.DataFrame([{"image_path": f"{label}_{i}.jpg", "label": label}
                              for label in cfg.CLASS_LABELS for i in range(5)])
        first = create_dataloaders(frame, None, None, num_workers=0)
        torch.randn(100)  # Unrelated model initialization must not alter sampler order.
        second = create_dataloaders(frame, None, None, num_workers=0)
        self.assertEqual(first[1].dataset.image_paths, second[1].dataset.image_paths)
        self.assertEqual(list(first[0].sampler), list(second[0].sampler))
        model = StudentModel(pretrained=False)
        baseline = create_optimizer(model, kd=False)
        distilled = create_optimizer(model, kd=True)
        self.assertEqual([g["lr"] for g in baseline.param_groups], [g["lr"] for g in distilled.param_groups])

    def test_training_loops_save_checkpoints_and_teacher_is_frozen(self):
        torch.manual_seed(59)
        loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.arange(8) % 7), batch_size=4)
        with tempfile.TemporaryDirectory() as folder:
            for kd in (False, True):
                model, teacher = nn.Linear(4, 7), nn.Linear(4, 7)
                teacher_before = {name: p.detach().clone() for name, p in teacher.named_parameters()}
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
                path = Path(folder) / f"{kd}.pth"
                stopper = EarlyStopping(2, path)
                common = (loader, loader, nn.CrossEntropyLoss(), optimizer, scheduler, "cpu", 1, stopper)
                history = fit_kd(model, teacher, *common) if kd else fit(model, *common)
                self.assertTrue(path.is_file())
                self.assertEqual(len(history[0]), 1)
                for name, p in teacher.named_parameters():
                    torch.testing.assert_close(p, teacher_before[name])
                    self.assertIsNone(p.grad)
            zero_path = Path(folder) / "zero.pth"
            EarlyStopping(2, zero_path)(1.0, 0.0, nn.Linear(4, 7))
            self.assertTrue(zero_path.is_file())


if __name__ == "__main__":
    unittest.main()
