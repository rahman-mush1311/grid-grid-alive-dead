import subprocess
import os
import platform
import shlex

def run_ffplay(video_path, width=None, height=None):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    # Build ffplay command
    cmd = ['ffplay', video_path]
    if width and height:
        cmd += ['-x', str(width), '-y', str(height)]

    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("ffplay not found. Make sure FFmpeg is installed and in your system PATH.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== FFplay Video Player ===")
    video_path = input("Enter full path to video file: ").strip()
    x_input = input("Enter window width (or press Enter to skip): ").strip()
    y_input = input("Enter window height (or press Enter to skip): ").strip()

    # Convert width/height to int if provided
    width = int(x_input) if x_input else 1920
    height = int(y_input) if y_input else 1080

    run_ffplay(video_path, width, height)
