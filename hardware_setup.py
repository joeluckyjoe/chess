import torch

def get_device():
    """
    Detects and returns the best available hardware device (TPU, GPU, or CPU).
    Also handles the installation of torch_xla for TPU support.
    """
    # Check for TPU
    try:
        import torch_xla.core.xla_model as xm
        print("✅ TPU detected. Using PyTorch/XLA.")
        return xm.xla_device()
    except ImportError:
        print("INFO: PyTorch/XLA not found. TPU not available.")
    except Exception as e:
        print(f"[WARNING] An error occurred while checking for TPU: {e}")

    # Check for GPU
    if torch.cuda.is_available():
        print("✅ GPU detected. Using CUDA.")
        return torch.device("cuda")

    # Default to CPU
    print("⚠️ No TPU or GPU detected. Defaulting to CPU.")
    return torch.device("cpu")

def install_xla_if_needed(device):
    """
    Installs the torch_xla library if the detected device is a TPU.
    This should be run in a Colab environment.
    """
    if 'xla' in str(device):
        try:
            import torch_xla
            print("PyTorch/XLA is already installed.")
        except ImportError:
            print("Installing PyTorch/XLA for TPU support...")
            # Use subprocess to run the pip install command
            import subprocess
            import sys
            command = [sys.executable, '-m', 'pip', 'install', 'torch~=2.3.0', 'torch_xla[tpu]~=2.3.0', '-f', 'https://storage.googleapis.com/libtpu-releases/index.html']
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print("--- XLA INSTALLATION FAILED ---")
                print(stderr.decode())
            else:
                print("--- XLA Installation Successful ---")
                print("Please restart the Colab runtime now for the changes to take effect.")
                # Exit to force user to restart runtime
                exit()