import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import re
import collections
import glob
import os 
import random
from sklearn.preprocessing import StandardScaler
from PIL import Image

import os
import shutil

def get_file_prefix(filename):
        if re.search(r"DeadObjectXYs\.txt", filename):
            return 'D', ''
        else:
            file_pattern = re.compile(r'''(\d{1,2}-\d{1,2}-\d{2})_(\w+)_ObjectXYs\.txt|AliveObjectXYs(\w+)\.txt''')
            match = file_pattern.search(filename)
            if match:
                if match.group(1)and match.group(2):
                    return (match.group(1), match.group(2)) 
                else: 
                    return ('AliveObjects',match.group(3))
            else:
                return '',''
        
def get_displacements(filelists):
    pattern = re.compile(r'''
        \s*(?P<object_id>\d+),
        \s*(?P<within_frame_id>\d+),
        \s*'(?P<file_path>[^']+)',
        \s*cX\s*=\s*(?P<x>\d+),
        \s*cY\s*=\s*(?P<y>\d+),
        \s*Frame\s*=\s*(?P<frame>\d+)
        ''', re.VERBOSE)
    frame_size_list=[]
    sample_size_list=[]
    for filename in filelists:
        observations = collections.defaultdict(list)
        #print(filename)
        prefix, extension = get_file_prefix(filename)
        
        frameCount=0
        seen_frames = set() 
        
        with open(filename) as object_xys:               
                for line in object_xys:
                    m = pattern.match(line)
                    if m:
                        obj_id = int(m.group('object_id'))
                        frame = int(m.group('frame'))
                        cX = int(m.group('x'))
                        cY = int(m.group('y'))
                        obj_id = obj_id
                        observations[obj_id].append((frame, cX, cY))
                        if frame not in seen_frames:
                            frameCount += 1
                            seen_frames.add(frame)
                        
        frame_size_list.append(frameCount)
        sample_size_list.append(len(observations))
        #print(f"from plot functions for {filename}: {frame_size_list,frameCount}")
        # Ensure observations are sorted by frame
        for object_id in observations:
            observations[object_id].sort()
            
        dataset_name=prefix+"_"+extension
       
        #print(f"from plot functions for {dataset_name}: {len(observations)}")
        plot_mean_covariance(observations,dataset_name)
    #print(f"final sample sizes & frame sizes:{sum(sample_size_list)} \n {sample_size_list} {frame_size_list} ")
    #plot_stat_bars(frame_size_list)
    
def plot_stat_bars(curr_list):

    # Already calculated values
    min_val = min(curr_list)
    avg_val = (sum(curr_list)/len(curr_list))
    max_val = max(curr_list)

    # Data and labels
    values = [min_val, avg_val, max_val]
    labels = ['Min', 'Average', 'Max']
    colors = ['skyblue', 'lightgreen', 'salmon']

    # Plot
    plt.bar(labels, values, color=colors)
    plt.title("Min, Average, and Max Values of Frame numbers in Alive Samples")
    plt.ylabel("Value")
    plt.ylim(0, max_val + 2)  # Add some padding above the max
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.show()
    
def plot_sample_sizes():
    

    values = [1239, 357]
    labels = ["Trainning Examples", "Testing Examples"]
    colors = ['red', 'blue']
    '''
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['skyblue','lightgreen'], startangle=140)
    plt.title("Training Vs Testing")
    plt.axis('equal')  # Equal aspect ratio ensures it's a circle
    '''
    plt.bar(labels, values, color=colors ,width=0.4)
    plt.title("Training Vs Testing")
    plt.ylabel("Sizes")
    
    plt.show()

def plot_mean_covariance(curr_obs,dataset_name):
    print(f"current {dataset_name} observation is len of: {len(curr_obs)}")
    
    points=[]
    
    displacements_dict={}
    
    for obj_id, obs in curr_obs.items():
            displacements=[]
            for i in range(len(obs) - 1):
                dframe = obs[i+1][0] - obs[i][0]
                #to do: dframe<=0 continue logging error
                if dframe>0:
                
                    dx = obs[i+1][1] - obs[i][1]
                    dy = obs[i+1][2] - obs[i][2]
                    points.append((dx/dframe,dy/dframe))
                    displacements.append((dx/dframe,dy/dframe))
                    
                else:
                    print(f"dframe has invalid {dframe}")
            displacements_dict[obj_id]=displacements
            #print(f"for {obj_id} observation size is {len(obs)} and displacements size: {len(displacements_dict[obj_id])}")
            
    print(f"displacements: {len(displacements_dict)}")
    
    for obj_id, dis in displacements_dict.items():
        max_dx=0
        max_dy=0
        for i in range(len(dis)):
            dx=dis[i][0]
            dy=dis[i][1]
            if dx>max_dx:
                max_dx=dx
            if dy>max_dy:
                max_dy=dy
        print(f"for {obj_id} x-axis maximum is {max_dx} y-axis maximum {max_dy}")
            
    '''
    points=np.array(points)
    # Compute the mean vector
    mean_vector = np.mean(points, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(points, rowvar=False)

    # Print results
    print("Mean Vector:\n", mean_vector)
    print("\nCovariance Matrix:\n", cov_matrix)
    
    scaler = StandardScaler()
    Z = scaler.fit_transform(points)  # Apply whitening

    # Step 2: Compute mean and covariance of transformed data
    Z_mean = np.mean(Z, axis=0)
    Z_cov = np.cov(Z, rowvar=False)

    # Display results
    print("Mean After Whitening:\n", Z_mean)
    print("\nCovariance Matrix After Whitening:\n", Z_cov)

    # Extract X and Y points for plotting
    X_points = points[:, 0]
    Y_points = points[:, 1]
    
    # Plot scatter points
    #fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.scatter(X_points, Y_points, color="blue", label="Data Points", edgecolors="black", s=100)
    
    # Plot mean as a red star
    ax.scatter(mean_vector[0], mean_vector[1], color="red", marker="*", s=20, label="Mean (μ)")

    # Compute eigenvalues and eigenvectors for covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Get the angle of the largest eigenvector (direction of most variance)
    angle = np.degrees(np.arctan2(*eigvecs[:, 1]))

    # Width and height of ellipse scaled by eigenvalues
    width, height = np.sqrt(eigvals)  

    # Create and add covariance ellipse
    ellipse = Ellipse(xy=mean_vector, width=width, height=height, angle=angle,
                  edgecolor='purple', facecolor='none', linestyle='--', linewidth=2, label="Covariance Ellipse")
    ax.add_patch(ellipse)

    # Labels and Formatting
    ax.set_title(f"{dataset_name} Plot with Mean and Covariance Ellipse Without Normalization")
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    
    
    
    # Plot mean as a red star
    axes[1].scatter(Z_mean[0], Z_mean[1], color="red", marker="*", s=20, label="Mean (μ)")

    # Compute eigenvalues and eigenvectors for covariance matrix
    eigvals_norm, eigvecs_norm = np.linalg.eigh(Z_cov)

    # Get the angle of the largest eigenvector (direction of most variance)
    angle_norm = np.degrees(np.arctan2(*eigvecs_norm[:, 1]))

    # Width and height of ellipse scaled by eigenvalues
    width_norm, height_norm = np.sqrt(eigvals_norm)  

    # Create and add covariance ellipse
    ellipse_norm = Ellipse(xy=Z_mean, width=width_norm, height=height_norm, angle=angle_norm,
                  edgecolor='purple', facecolor='none', linestyle='--', linewidth=2, label="Covariance Ellipse")
    axes[1].add_patch(ellipse_norm)

    # Labels and Formatting
    axes[1].set_title(f"{dataset_name} Plot with Mean and Covariance Ellipse With Normalization")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Y-axis")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    '''
    
def mean_covariance_plot(grid_mu,grid_cov):
    # Step 1: Compute global range for all plots
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')

    # Iterate through the grid to find the global range
    for i, (mu_row_item,cov_row_item) in enumerate(zip(grid_mu,grid_cov)):
        for j, (mu_col_item, cov_col_item) in enumerate(zip(mu_row_item, cov_row_item)):
            mu = mu_col_item
            cov_matrix = cov_col_item
            #print(f"for {i} {j} {mu} \n,{cov_matrix}")
    
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            width, height = 2 * np.sqrt(eigenvalues)
            max_range = max(width, height) * 1.5
            
            # Update global min and max values
            global_min_x = min(global_min_x, mu[0] - max_range)
            global_max_x = max(global_max_x, mu[0] + max_range)
            global_min_y = min(global_min_y, mu[1] - max_range)
            global_max_y = max(global_max_y, mu[1] + max_range)
            
    print(global_max_y,global_max_x)
    print(global_min_y,global_max_y)
    
    for i, (mu_row_item,cov_row_item) in enumerate(zip(grid_mu,grid_cov)):
        for j, (mu_col_item, cov_col_item) in enumerate(zip(mu_row_item, cov_row_item)):
            mu = mu_col_item
            cov_matrix = cov_col_item
            
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot the mean as a point
            ax.plot(mu[0], mu[1], 'ro', label="Mean")
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Sort eigenvalues and eigenvectors
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    
            # Calculate the angle of the ellipse
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

            # Width and height of the ellipse based on n_std standard deviations
            width, height = 2 * 1.0 * np.sqrt(eigenvalues)

            # Draw the ellipse
            ellipse_1std = Ellipse(xy=mu, width=width, height=height, angle=angle,edgecolor='blue', linestyle='--', linewidth=2, facecolor='none', label="1 Std Dev")
            ax.add_patch(ellipse_1std)
            
            print(global_max_y,global_max_x)
            print(global_min_y,global_max_y)
            
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_y, global_max_y)

            # Set equal aspect ratio for both axes
            ax.set_aspect('equal', adjustable='datalim')
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True)
            ax.set_title(f"Visualization of Mean and Covariance Matrix as Ellipses Alive Grid Stats{[i]}{[j]}")
            #plt.axis('equal')
            plt.savefig(f"alive_model_grid_stats[{i}][{j}]")
            plt.show()
            
def make_collage():


    # Folder containing your saved images
    image_folder = r"D:\RA work Fall2024\grid-grid-alive-dead\outlier_grid_stats"
    output_file = r"alive_grid_stat_collage.png"

    # Get list of image files
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]
    print(len(image_files))


    # Load all images and get their dimensions
    images = [Image.open(img) for img in image_files]
    img_width, img_height = images[0].size

    # Define grid size (6x5 for 26 images)
    columns, rows = 5,5
    collage_width = columns * img_width
    collage_height = rows * img_height

    # Create blank canvas for the collage
    collage = Image.new("RGB", (collage_width, collage_height), (255, 255, 255))

    # Paste each image into the collage
    for idx, img in enumerate(images):
        x_offset = (idx % columns) * img_width
        y_offset = (idx // columns) * img_height
        collage.paste(img, (x_offset, y_offset))

    # Save the final collage
    collage.save(output_file)
    print(f"Collage saved as {output_file}")                
if __name__ == "__main__":
    
    #source_folder = r'''D:\RA work Fall2024\january data'''
    #destination_folder = r'''D:\RA work Fall2024\grid-grid-alive-dead\organic_files'''
    '''
    # Make sure destination exists
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all files in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".txt"):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(destination_folder, file)

                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                    print(f"Moved: {src_path} → {dst_path}")
                else:
                    print(f"Skipped (already exists): {dst_path}")
    
    file_list = []#'AliveObjectXYs1at.txt'
    
    subfolder = "dead_files"  
    for root, dirs, files in os.walk(subfolder):
        for file in files:
            if file.endswith(".txt"):  # Only add .txt files
                file_list.append(os.path.join(root, file))
    print(len(file_list))
    '''
    
    plot_sample_sizes()
    #'AliveObjectXYs2at.txt','AliveObjectXYs3at.txt','AliveObjectXYs4at.txt','AliveObjectXYs5at.txt','AliveObjectXYs6at.txt','AliveObjectXYs8at.txt']
    #'12-27-24_1a_ObjectXYs.txt','12-27-24_1b_ObjectXYs.txt','1-3-25_1a_ObjectXYs.txt','1-6-25_1a_ObjectXYs.txt'
    #get_displacements(file_list)
    #make_collage()
