import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, shapiro
from matplotlib.patches import Ellipse
import re
import collections
import glob
import os 
import random
from sklearn.preprocessing import StandardScaler
from PIL import Image
import seaborn as sns
import os
import shutil

LOG_PDFS='log_pdfs'

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
 
    #observations = collections.defaultdict(list)
    
    for filename in filelists:
        observations = collections.defaultdict(list)
        dead_observations = collections.defaultdict(list)
        alive_observations = collections.defaultdict(list)
        print(filename)
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
                        obj_id = f"{prefix}_{obj_id}_{extension}"
                        observations[obj_id].append((frame, cX, cY))
                       
        # Ensure observations are sorted by frame
        for object_id in observations:
            observations[object_id].sort()
            
        dataset_name=prefix+"_"+extension
       
        #plot_mean_covariance(observations,dataset_name)
        dead_observations,alive_observations=split_observations_by_displacements(observations)
    
    
def plot_stat_bars(curr_list):

    # Already calculated values
    #min_val = min(curr_list)
    #avg_val = (sum(curr_list)/len(curr_list))
    #max_val = max(curr_list)
    

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

def plot_histogram(curr_dead_with_dead, curr_alive_with_dead):

    alive_pdf_list=[]   
    dead_pdf_list=[]
    
    for obj_id, pdfs in curr_dead_with_dead.items():
        curr_pdfs=curr_dead_with_dead[obj_id][LOG_PDFS]
        dead_pdf_list.extend(curr_pdfs)
    
    for obj_id, pdfs in curr_alive_with_dead.items():
        curr_pdfs=curr_alive_with_dead[obj_id][LOG_PDFS]
        alive_pdf_list.extend(curr_pdfs)
    
    print(len(alive_pdf_list), len(dead_pdf_list))
    
    dead_sorted_values = np.sort(dead_pdf_list)
    alive_sorted_values = np.sort(alive_pdf_list)
    # Step 2: Calculate the cumulative probabilities
    dead_cdf = np.arange(1, len(dead_sorted_values) + 1) / len(dead_sorted_values)
    alive_cdf = np.arange(1, len(alive_sorted_values) + 1) / len(alive_sorted_values)
    
    print("Range of PDF 1:", dead_sorted_values.min(), "to", dead_sorted_values.max())
    print("Range of PDF 2:", alive_sorted_values.min(), "to", alive_sorted_values.max())
    

    # Step 3: Plot the CDFs side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CDF for the first set of probability density values
    axes[0].plot(dead_sorted_values, dead_cdf,color='blue')
    axes[0].set_title('CDF of Dead PDF', fontsize=14)
    axes[0].set_xlabel('Probability Density Log Values Dead With Dead', fontsize=12)
    axes[0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[0].grid()

    # CDF for the second set of probability density values
    axes[1].plot(alive_sorted_values, alive_cdf,color='green')
    axes[1].set_title('CDF of Alive PDF ', fontsize=14)
    axes[1].set_xlabel('Probability Density Log Values Alive With Dead', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].grid()
    
    #plt.savefig(f"dead_alive_cdf_line(n).png", format='png', dpi=300)
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

   
    plt.show()
   
def plot_sample_sizes():
    

    values = [211, 83, 1]
    labels = ["Max", "Average", "Min"]
    colors = ['skyblue', 'lightgreen', 'salmon']
    '''
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['skyblue','lightgreen'], startangle=140)
    plt.title("Training Vs Testing")
    plt.axis('equal')  # Equal aspect ratio ensures it's a circle
    '''
    plt.bar(labels, values, color=colors ,width=0.4)
    plt.title("Displacements(Movement Sequence) Lengths")
    plt.ylabel("Sizes")
    
    plt.show()

def plot_mean_covariance(curr_obs,dataset_name):
    print(f"current {dataset_name} observation is len of: {len(curr_obs)}")
    
    dx_points=[]
    dy_points=[]
    distances=[]
    points=[]
    
    
    for obj_id, obs in curr_obs.items():
        #print(obj_id)
        for i in range(len(obs) - 1):
            if len(obs)>3:
                dframe = obs[i+1][0] - obs[i][0]                   
                if dframe>0:
                        
                    dx = obs[i+1][1] - obs[i][1]
                    dy = obs[i+1][2] - obs[i][2]
                    points.append((dx/dframe,dy/dframe))
                    if obj_id=='AliveObjects_143_1at':
                        #print("found it")
                        dx_points.append(obs[i][1])
                        dy_points.append(obs[i][2])                    
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
    
    sns.jointplot(x=points[:, 0], y=points[:, 1], kind='kde')
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.title(f"Displacement Distribution for {dataset_name}")
    plt.show()
   
  
    
def split_observations_by_displacements(curr_obs):
       
    all_dx = []
    all_dy = []
    object_avg = {}
    
    dead_obs = collections.defaultdict(list)
    alive_obs = collections.defaultdict(list)
    # First pass: compute max dx/dy per object
    for obj_id, obs in curr_obs.items():
        dx_list = []
        dy_list = []
        for i in range(len(obs) - 1):
            dframe = obs[i+1][0] - obs[i][0]
            if dframe > 0:
                dx = (obs[i+1][1] - obs[i][1]) / dframe
                dy = (obs[i+1][2] - obs[i][2]) / dframe
                dx_list.append(dx)
                dy_list.append(dy)
                all_dx.append(dx)
                all_dy.append(dy)
                    
            else:
                print(f"dframe has invalid value: {dframe}")

        if dx_list and dy_list:
            avg_dx = sum(dx_list) / len(dx_list)
            avg_dy = sum(dy_list) / len(dy_list)
            object_avg[obj_id] = (avg_dx, avg_dy)

    # Global averages of dx and dy across all objects
    global_avg_dx = sum(all_dx) / len(all_dx)
    global_avg_dy = sum(all_dy) / len(all_dy)

    #print(f"Global avg max dx: {global_avg_dx:.2f}, dy: {global_avg_dy:.2f}")

    # Second pass: classify based on max dx/dy vs global averages
    for obj_id, (avg_dx, avg_dy) in object_avg.items():
        obs = curr_obs[obj_id]
        #print(f"for {obj_id}: {max_dx}, {max_dy}")
        if len(obs) > 5 and (avg_dx > global_avg_dx and avg_dy > global_avg_dy):
            alive_obs[obj_id] = len(obs)
        elif len(obs) > 5:
            dead_obs[obj_id] = len(obs)

    print(f"From split function: total={len(curr_obs)}, dead={len(dead_obs)}, alive={len(alive_obs)}")
    dead_total_sum = sum(dead_obs.values()) 
    alive_total_sum=sum(alive_obs.values())
    print(f"dead_ds {dead_total_sum}, alive_ds {alive_total_sum}")
    return dead_obs, alive_obs   
    
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
            ax.set_title(f"Visualization of Mean and Covariance Matrix{[i]}{[j]}")
            #plt.axis('equal')
            plt.savefig(f"dead_model_grid_stats[{i}][{j}]")
            #plt.show()
            
def make_collage():


    # Folder containing your saved images
    image_folder = r"D:\RA work Fall2024\grid-grid-alive-dead\results\outlier_grid_pictures"
    output_file = r"outlier_rid_stat_collage.pdf"

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
    collage.save(output_file, "PDF", resolution=300.0)
    print(f"Collage saved as {output_file}")
    
def mean_covariance_overlay_plot(grid_mu_alive, grid_cov_alive, grid_mu_dead, grid_cov_dead):
    # Step 1: Compute global min/max across both alive and dead models
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')

    all_models = [(grid_mu_alive, grid_cov_alive), (grid_mu_dead, grid_cov_dead)]

    for grid_mu, grid_cov in all_models:
        for i, (mu_row_item, cov_row_item) in enumerate(zip(grid_mu, grid_cov)):
            for j, (mu_col_item, cov_col_item) in enumerate(zip(mu_row_item, cov_row_item)):
                mu = mu_col_item
                cov_matrix = cov_col_item

                eigenvalues, _ = np.linalg.eigh(cov_matrix)
                width, height = 2 * np.sqrt(eigenvalues)
                max_range = max(width, height) * 1.5

                global_min_x = min(global_min_x, mu[0] - max_range)
                global_max_x = max(global_max_x, mu[0] + max_range)
                global_min_y = min(global_min_y, mu[1] - max_range)
                global_max_y = max(global_max_y, mu[1] + max_range)

    # Step 2: Plot overlay for each grid cell
    for i in range(len(grid_mu_alive)):
        for j in range(len(grid_mu_alive[0])):
            mu_alive = grid_mu_alive[i][j]
            cov_alive = grid_cov_alive[i][j]
            mu_dead = grid_mu_dead[i][j]
            cov_dead = grid_cov_dead[i][j]

            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot alive mean
            ax.plot(mu_alive[0], mu_alive[1], 'go', label="Alive Mean")
            eigenvalues, eigenvectors = np.linalg.eigh(cov_alive)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse_alive = Ellipse(xy=mu_alive, width=width, height=height, angle=angle,
                                    edgecolor='green', linestyle='--', linewidth=2, facecolor='none', label="Alive 1 Std Dev")
            ax.add_patch(ellipse_alive)

            # Plot dead mean
            ax.plot(mu_dead[0], mu_dead[1], 'ro', label="Dead Mean")
            eigenvalues, eigenvectors = np.linalg.eigh(cov_dead)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse_dead = Ellipse(xy=mu_dead, width=width, height=height, angle=angle,
                                   edgecolor='red', linestyle='-', linewidth=2, facecolor='none', label="Dead 1 Std Dev")
            ax.add_patch(ellipse_dead)

            # Set plot limits and labels
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_y, global_max_y)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            ax.set_title(f"Alive vs Dead Covariance Ellipses Grid [{i}][{j}]")
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"alive_dead_overlay_grid_stats[{i}][{j}].png")
            #plt.show()
            
def plot_accuracy_window():
    '''
    window_sizes = list(range(1, 11))  # Window sizes from 1 to 10
    #accuracy_values = [0.772, 0.770, 0.777, 0.774, 0.773, 0.773, 0.773, 0.773, 0.773, 0.775] 
    #accuracy_values = [-19.088, -10.104, -19.088, -19.088, -19.088, -19.088, -19.088, -19.088, -19.088, -19.088] 
    accuracy_values = [0.392, 0.425, 0.562, 0.455, 0.375, 0.333, 0.250, 0.000, 0.000, 0.000] 
    '''
    grid_sizes = [3, 5]
    parameter_counts = [grid**2 * 5 for grid in grid_sizes]
    sample_counts = [271, 1300]
    
    outlier_train_accuracy = [0.661, 0.726]   
    bayesian_train_accuracy = [0.638, 0.767]
    outlier_test_accuracy = [0.638, 0.717]   
    bayesian_test_accuracy = [0.582, 0.737]
    
    x_labels = [f"{p} ({s})" for p, s in zip(parameter_counts, sample_counts)]
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, outlier_train_accuracy, label='Outlier Train Accuracy', linestyle='-', color='blue')
    plt.plot(x_labels, bayesian_train_accuracy, label='Bayesian Train Accuracy', linestyle='-', color='red')
    plt.plot(x_labels, outlier_test_accuracy, label='Outlier Test Accuracy', linestyle='-', color='green')
    plt.plot(x_labels, bayesian_test_accuracy, label='Bayesian Test Accuracy', linestyle='-', color='orange')
   

    plt.xlabel('Number of Parameters (Number of Samples)')
    plt.ylabel('Accuracy')
    plt.title('Outlier vs Bayesian Model Accuracy Across Grid Sizes')
    plt.grid(True)
    plt.ylim(0.50, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

          
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
    '''
    file_list = []
    i=0
    flag=0
    subfolder = "alive_files"  
    for root, dirs, files in os.walk(subfolder):
        for file in files:
            if file.endswith(".txt"):  # Only add .txt files
                file_list.append(os.path.join(root, file))
                i+=1
            if i>=5:
                flag=1
                break
        if flag==1:
            break
    print(len(file_list),file_list)
    
    
    #plot_sample_sizes()
    #'AliveObjectXYs2at.txt','AliveObjectXYs3at.txt','AliveObjectXYs4at.txt','AliveObjectXYs5at.txt','AliveObjectXYs6at.txt','AliveObjectXYs8at.txt']
    #'12-27-24_1a_ObjectXYs.txt','12-27-24_1b_ObjectXYs.txt','1-3-25_1a_ObjectXYs.txt','1-6-25_1a_ObjectXYs.txt'
    #get_displacements(file_list)
    #make_collage()
    plot_accuracy_window()
