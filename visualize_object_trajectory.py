import subprocess
import os
import platform
import shlex
import numpy
import matplotlib.pyplot as plt

def run_ffplay(video_path, width, height):
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

def plot_one_object(curr_obs,target_obj_id,filename):

    """
    Plots one object trajectory using x/y limits based on all objects combined.
    
    Params:
        - observations: dict of {object_id: [(frame, x, y), ...]}
        - all_ids: list of all relevant object IDs (subset of keys in observations)
        - object_to_plot: str, one object ID to visualize
    """
    if target_obj_id not in observations:
        print(f"Object {target_obj_id} not found.")
        return

    # Step 1: Compute global x/y bounds from all_ids
    all_x = []
    all_y = []
    for obj_id in all_ids:
        if obj_id in observations:
            all_x.extend(p[1] for p in observations[obj_id])
            all_y.extend(p[2] for p in observations[obj_id])

    if not all_x or not all_y:
        print("No valid coordinates found in provided object list.")
        return

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Step 2: Extract trajectory of the target object
    traj = observations[object_to_plot]
    traj_x = [p[1] for p in traj]
    traj_y = [p[2] for p in traj]

    # Step 3: Plot
    plt.figure(figsize=(7, 6))
    plt.plot(traj_x, traj_y, marker='o', linestyle='-', color='blue', label=f'Object: {target_obj_id}')
    plt.title(f"Trajectory of {object_to_plot}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
def plot_test_objects(curr_obs, curr_obs_pred):
    print("=== Yet to work ===")    
    
def take_user_input():
    print("=== FFplay Video Player ===")
    video_path = input("Enter full path to video file: ").strip()
    x_input = input("Enter window width (or press Enter to skip): ").strip()
    y_input = input("Enter window height (or press Enter to skip): ").strip()

    # Convert width/height to int if provided
    width = int(x_input) if x_input else 1920
    height = int(y_input) if y_input else 1080

    run_ffplay(video_path, width, height)

    
