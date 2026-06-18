import platform
import shutil
import subprocess
import sys


def section(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run_command(command: list[str]):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError:
        print(f"{command[0]} not found")
        return
    except Exception as exc:
        print(f"Failed to run {' '.join(command)}: {type(exc).__name__}: {exc}")
        return

    output = (result.stdout or result.stderr or "").strip()
    if output:
        print(output)
    else:
        print(f"No output. Exit code: {result.returncode}")


def check_python():
    section("Python")
    print(f"Executable: {sys.executable}")
    print(f"Version:    {sys.version}")
    print(f"Platform:   {platform.platform()}")


def check_nvidia_smi():
    section("NVIDIA Driver")
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found.")
        print("This usually means NVIDIA driver is missing, not installed correctly, or not on PATH.")
        return

    run_command(["nvidia-smi"])
    print()
    print("GPU summary:")
    run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )


def check_torch():
    section("PyTorch CUDA")
    try:
        import torch
    except Exception as exc:
        print(f"torch import failed: {type(exc).__name__}: {exc}")
        return

    print(f"torch version:       {torch.__version__}")
    print(f"torch cuda build:    {torch.version.cuda}")
    print(f"cuda available:      {torch.cuda.is_available()}")
    print(f"cuda device count:   {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            memory_gb = props.total_memory / 1024**3
            print(f"device {index}:          {torch.cuda.get_device_name(index)} ({memory_gb:.1f} GB)")
    else:
        print("PyTorch is not using CUDA.")
        print("If you have an NVIDIA GPU, install a CUDA-enabled torch build.")


def check_onnxruntime():
    section("ONNX Runtime Providers")
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"onnxruntime import failed: {type(exc).__name__}: {exc}")
        return

    print(f"onnxruntime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"available providers: {providers}")

    if "CUDAExecutionProvider" in providers:
        print("ONNX Runtime GPU provider is available.")
    else:
        print("CUDAExecutionProvider is not available.")
        print("Install onnxruntime-gpu if you want InsightFace to use GPU.")


def check_transformers_device_hint():
    section("Photo Sorter GPU Readiness")
    try:
        import torch

        torch_gpu = torch.cuda.is_available()
    except Exception:
        torch_gpu = False

    try:
        import onnxruntime as ort

        ort_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        ort_gpu = False

    print(f"DETR / CLIP will use: {'GPU' if torch_gpu else 'CPU'}")
    print(f"InsightFace will use: {'GPU preferred' if ort_gpu else 'CPU'}")

    if not torch_gpu or not ort_gpu:
        print()
        print("Suggested installs for NVIDIA GPU:")
        print("1. Install or update NVIDIA driver.")
        print("2. Install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/")
        print("3. Replace onnxruntime with onnxruntime-gpu:")
        print("   pip uninstall onnxruntime")
        print("   pip install onnxruntime-gpu")


def main():
    check_python()
    check_nvidia_smi()
    check_torch()
    check_onnxruntime()
    check_transformers_device_hint()


if __name__ == "__main__":
    main()
