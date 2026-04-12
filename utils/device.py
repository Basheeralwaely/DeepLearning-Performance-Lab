"""Device detection and information utilities.

Provides helpers for selecting the best available compute device and
printing detailed hardware information, so tutorials can adapt to
whatever hardware is available.
"""

import logging

import torch


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Returns CUDA device if a GPU is available, otherwise falls back to CPU.
    Logs the selected device for visibility in tutorial output.

    Returns:
        torch.device configured for CUDA or CPU.
    """
    logger = logging.getLogger("utils.device")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Using GPU: %s", gpu_name)
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")

    return device


def print_device_info() -> None:
    """Print detailed information about the available compute device.

    When CUDA is available, prints GPU name, total memory, and CUDA version.
    Otherwise, prints basic CPU information.
    """
    print()
    print("=" * 50)
    print(" Device Information")
    print("=" * 50)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()

        print(f"  GPU:           {gpu_name}")
        print(f"  GPU Memory:    {gpu_mem:.1f} GB")
        print(f"  CUDA Version:  {cuda_version}")
        print(f"  GPU Count:     {device_count}")
    else:
        import platform

        print(f"  Device:        CPU")
        print(f"  Platform:      {platform.processor() or platform.machine()}")

    print(f"  PyTorch:       {torch.__version__}")
    print("=" * 50)
    print()
