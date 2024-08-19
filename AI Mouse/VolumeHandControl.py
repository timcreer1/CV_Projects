import subprocess

def set_volume(volume):
    # Normalize the volume to be between 0 and 100
    normalized_volume = max(0, min(100, int(volume * 100)))

    # Set the system volume
    subprocess.run(["osascript", "-e", f"set volume output volume {normalized_volume}"])

# Example: Set the volume to 50%
set_volume(0.5)