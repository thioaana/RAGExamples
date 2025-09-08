import subprocess

def check_gpu():
    # Try nvidia-smi
    try:
        output = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.STDOUT
        ).decode()
        first_line = output.split("\n")[0]
        print("✅ GPU detected via nvidia-smi")
        print(first_line)
        return True
    except Exception:
        pass

    # Try numba if installed
    try:
        from numba import cuda
        if cuda.is_available():
            print("✅ GPU detected via numba:", cuda.get_current_device().name)
            return True
    except Exception:
        pass

    print("❌ No GPU detected")
    return False