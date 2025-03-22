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

def get_file_prefix(filename):
        if re.search(r"DeadObjectXYs\.txt", filename):
            return 'D', ''
        else:
            file_pattern = re.compile(r'''(\d{1,2}-\d{1,2}-\d{2})_(\d+)_ObjectXYs\.txt|AliveObjectXYs(\d+)([a-z])\.txt''')
            match = file_pattern.search(filename)
            if match.group(1)and match.group(2):
                return (match.group(1), match.group(2)) 
            else:
                return (match.group(3),match.group(4))
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
        
    for filename in filelists:
        observations = collections.defaultdict(list)
        prefix, extension = get_file_prefix(filename)
        
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
        
        # Ensure observations are sorted by frame
        for object_id in observations:
            observations[object_id].sort()
            
        dataset_name=prefix+extension
        plot_mean_covariance(observations,dataset_name)
        #print(f"from plot functions: {len(observations)}")
        
def plot_mean_covariance(curr_obs,dataset_name):
    print(f"current {dataset_name} observation is len of: {len(curr_obs)}")
    
    points=[]
    
    for obj_id, obs in curr_obs.items():
            for i in range(len(obs) - 1):
                dframe = obs[i+1][0] - obs[i][0]
                #to do: dframe<=0 continue logging error
                if dframe>0:
                
                    dx = obs[i+1][1] - obs[i][1]
                    dy = obs[i+1][2] - obs[i][2]
                    points.append((dx/dframe,dy/dframe))
                else:
                    print(f"dframe has invalid {dframe}")
    
    
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #ax.scatter(X_points, Y_points, color="blue", label="Data Points", edgecolors="black", s=100)
    
    # Plot mean as a red star
    axes[0].scatter(mean_vector[0], mean_vector[1], color="red", marker="*", s=20, label="Mean (μ)")

    # Compute eigenvalues and eigenvectors for covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Get the angle of the largest eigenvector (direction of most variance)
    angle = np.degrees(np.arctan2(*eigvecs[:, 1]))

    # Width and height of ellipse scaled by eigenvalues
    width, height = np.sqrt(eigvals)  

    # Create and add covariance ellipse
    ellipse = Ellipse(xy=mean_vector, width=width, height=height, angle=angle,
                  edgecolor='purple', facecolor='none', linestyle='--', linewidth=2, label="Covariance Ellipse")
    axes[0].add_patch(ellipse)

    # Labels and Formatting
    axes[0].set_title(f"{dataset_name} Plot with Mean and Covariance Ellipse Without Normalization")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    
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
