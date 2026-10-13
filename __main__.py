"""Run the project with python -m ISIC2018 <command> [options]."""

import argparse
from importlib import import_module
import runpy
import sys


COMMANDS = {
    "benchmark": ".src.benchmarking.cli",
    "export": ".src.inference.export",
    "build-engine": ".src.inference.build_engine",
    "predict": ".src.inference.predict",
    "evaluate": ".src.training.evaluation",
    "check-checkpoint": ".diagnostics.check_checkpoint",
}
DIAGNOSTICS = {
    "check-env": ".diagnostics.check_environment",
    "check-gpu": ".diagnostics.check_gpu",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="ISIC2018 training, deployment, and benchmarking",
        epilog="Use python -m ISIC2018 <command> --help for command options.",
    )
    parser.add_argument("command", choices=[*COMMANDS, "train", "download", *DIAGNOSTICS])
    if not argv:
        parser.print_help()
        return
    command = parser.parse_args(argv[:1]).command
    options = argv[1:]
    if command in COMMANDS:
        import_module(COMMANDS[command], package=__package__).main(options)
    elif command == "train":
        training = argparse.ArgumentParser(description="Train a teacher, student, or KD student")
        training.add_argument("kind", choices=("teacher", "student", "student-kd"))
        kind = training.parse_args(options[:1]).kind.replace("-", "_")
        import_module(".src.training.experiment", package=__package__).main(kind, options[1:])
    else:
        # Commands without configuration still support --help and reject extra arguments.
        argparse.ArgumentParser(description=f"ISIC2018 {command}").parse_args(options)
        module = ".src.data.download" if command == "download" else DIAGNOSTICS[command]
        runpy.run_module(__package__ + module, run_name="__main__")


if __name__ == "__main__":
    main()
