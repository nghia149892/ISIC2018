# ISIC2018 skin lesion classification

DINOv2 teacher, ResNet18 student, knowledge distillation (KD), and deployment
with ONNX / standalone TensorRT. Task 3 classifies seven skin lesion categories;
this is not a binary cancer/non-cancer classifier.
## Project structure

The source layout below describes the project workflow. Training data, model
weights, and generated results are local files, not required repository contents.

```text
ISIC2018/
├── src/
│   ├── config.py                 # Paths, class order, training defaults
│   ├── checkpoints.py            # Shared checkpoint selection and loading
│   ├── data/
│   │   ├── labels.py             # Read/validate ground truth and discover images
│   │   ├── transforms.py         # Shared validation/inference preprocessing
│   │   ├── dataset.py            # Dataset and stratified train/validation split
│   │   └── download.py           # Download training data
│   ├── models/
│   │   ├── classifiers.py        # DINOv2 teacher and torchvision ResNet18 student
│   │   └── resnet.py             # Educational ResNet implementation
│   ├── training/
│   │   ├── experiment.py         # Common training CLI and experiment manifest
│   │   ├── loops.py              # Supervised and KD training loops
│   │   ├── optimization.py       # Optimizer, LR schedule, early stopping
│   │   ├── evaluation.py         # Validation/test evaluation
│   │   ├── reproducibility.py    # Seeds
│   │   └── plots.py              # Optional visualizations
│   ├── inference/
│   │   ├── export.py             # Checkpoint → ONNX and numerical verification
│   │   ├── build_engine.py       # ONNX → TensorRT engine
│   │   ├── backends.py           # Common backend interface and ORT sessions
│   │   ├── predict.py            # Single-image prediction
│   │   └── tensorrt_runtime.py   # Load/run a standalone .engine
│   └── benchmarking/
│       ├── cli.py                # Arguments and orchestration
│       ├── runner.py             # Warmup and timing
│       ├── metrics.py            # Accuracy, macro-F1, recall, latency, agreement
│       └── report.py             # Console and JSON reports
├── __main__.py                   # Unified python -m ISIC2018 CLI
├── diagnostics/                  # Environment, GPU, checkpoint checks
├── tests/                        # Workflow regression tests
├── examples/                     # Incomplete historical encryption example
├── requirements.txt              # Core Python dependencies
├── requirements-tensorrt.txt      # Optional standalone TensorRT dependencies
├── ISIC2018_Task3_Test_Input/     # Local test images
├── ISIC2018_Task3_Test_GroundTruth/ # Local test labels
├── data/                         # Local training images/CSV (created on download)
├── weights/                      # Local checkpoints, ONNX, TensorRT engines
└── results/                      # Local generated reports / verification artifacts
```

