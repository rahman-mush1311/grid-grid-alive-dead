import re
import collections
import glob
import os 
import random
import numpy

DEAD='d'
ALIVE='a'

class ParsingObservations:
    def __init__(self):
        self.filelists = []
    
    def load_files_from_folder(self,filetype,filecount):
        '''
        in here we get the list of files/datasets from a particular folders (DEAD/ALIVE folders)
        Parameters:
        -filetype: which subfolder to go to
        -filecount: int how many files/datasets to consider
        '''
        counter_flag=0
        
        if filetype==DEAD:
            subfolder = "dead_files"  
        else:
            subfolder = "alive_files"
            
        for root, dirs, files in os.walk(subfolder):
            for file in files:
                if file.endswith(".txt"):  # Only add .txt files
                    self.filelists.append(os.path.join(root, file))
                    
                if len(self.filelists)>=filecount: #this is to give how many files to work with
                    counter_flag=1
                    break
            if counter_flag==1:
                break
        #print(self.filelists)            
    def load_observations(self,filenames):
        """
        Processes the input files and parses them to extract object ID, frame, x, and y coordinates.
        Parameters:
        -filenames: list of filenames to parse
        Returns:
        -dead_observations: a dictionary (object id: (frame,x_cordinate,y_coordinate)).
        -alive_observations: a dictionary (object id: (frame,x_cordinate,y_coordinate))
        """
        pattern = re.compile(r'''
        \s*(?P<object_id>\d+),
        \s*(?P<within_frame_id>\d+),
        \s*'(?P<file_path>[^']+)',
        \s*cX\s*=\s*(?P<x>\d+),
        \s*cY\s*=\s*(?P<y>\d+),
        \s*Frame\s*=\s*(?P<frame>\d+)
        ''', re.VERBOSE)
        
        
        
        for filename in filenames:
            prefix, extension = self.get_file_prefix(filename)
            print(prefix,extension)
            observations = collections.defaultdict(list)
            dead_observations = collections.defaultdict(list)
            alive_observations = collections.defaultdict(list)
            
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
                        
            print(f"from observation parser filename is: {filename}: observations size: {len(observations)}")
            dead_observations, alive_observations=self.split_observations_by_displacements(observations,prefix)
            print(f"From observation function: total={len(observations)}, dead={len(dead_observations)}, alive={len(alive_observations)}")
        # Ensure observations are sorted by frame
        '''
        for object_id in observations:
            observations[object_id].sort()
        
        for object_id, items in observations.items():
            assert all(items[i][0] <= items[i + 1][0] for i in range(len(items) - 1)), f"Items for {object_id} are not sorted by frame"
        '''
        
        for object_id in dead_observations:
            dead_observations[object_id].sort()
        
        for object_id, items in dead_observations.items():
            assert all(items[i][0] <= items[i + 1][0] for i in range(len(items) - 1)), f"Items for {object_id} are not sorted by frame"
        
        for object_id in alive_observations:
            dead_observations[object_id].sort()
        
        for object_id, items in alive_observations.items():
            assert all(items[i][0] <= items[i + 1][0] for i in range(len(items) - 1)), f"Items for {object_id} are not sorted by frame"
        
        
        return dead_observations,alive_observations
    
    def get_file_prefix(self, filename):
        '''
        extract the filename/dataset name to append it to object_id, since each dataset starts with 1... appending to same dictionaries will cause issues.
        Parameters:
        -filename: a str containing dataset/filename
        Returns:
        str matching re patterns
        '''
        if re.search(r"DeadObjectXYs\.txt", filename):
            return 'D', ''
        else:
            file_pattern = re.compile(r'''(\d{1,2}-\d{1,2}-\d{2})_(\d+)_ObjectXYs\.txt|AliveObjectXYs(\w+)\.txt''')
            match = file_pattern.search(filename)
            if match.group(1)and match.group(2):
                return (match.group(1), match.group(2)) 
            else:
                return ('Alive',match.group(3))
        return '',''
    
    def split_observations_by_displacements(self, curr_obs,prefix):
       
        all_dx_dy = []
        object_avg = {}
        dxdy_stats = {}
        dead_obs = collections.defaultdict(list)
        alive_obs = collections.defaultdict(list)
        
        # First pass: compute max dx/dy per object
        for obj_id, obs in curr_obs.items():
            curr_obj_dxdy=[]
                
            for i in range(len(obs) - 1):
                dframe = obs[i+1][0] - obs[i][0]
                if dframe > 0:
                    dx = (obs[i+1][1] - obs[i][1]) / dframe
                    dy = (obs[i+1][2] - obs[i][2]) / dframe
                    curr_obj_dxdy.append([dx,dy])                        
                    all_dx_dy.append([dx,dy])
                    
                else:
                    print(f"dframe has invalid value: {dframe}")

            if curr_obj_dxdy:
                curr_obj_dxdy_np = numpy.array(curr_obj_dxdy)
                curr_obj_mu = numpy.mean(curr_obj_dxdy_np , axis=0)
               
                avg_dx,avg_dy=curr_obj_mu[0],curr_obj_mu[1]
                object_avg[obj_id] = (avg_dx, avg_dy)
              
            # Global averages of dx and dy across all objects
        all_dx_dy_np=numpy.array(all_dx_dy)
        all_dx_dy_mu=numpy.mean(all_dx_dy_np, axis=0)                       
        global_avg_dx,global_avg_dy = all_dx_dy_mu[0],all_dx_dy_mu[1]
        
        #print(f"Global avg max dx: {global_avg_dx:.2f}, dy: {global_avg_dy:.2f}")
        
            # Second pass: classify based on global averages
        for obj_id, (avg_dx, avg_dy) in object_avg.items():
            obs = curr_obs[obj_id]
            if len(obs) > 5 and (avg_dx > global_avg_dx and avg_dy > global_avg_dy):
                alive_obs[obj_id] = obs
            elif len(obs) > 5:
                dead_obs[obj_id] = obs
     

        
        return dead_obs,alive_obs
    
    def prepare_train_test(self,curr_obs,train_ratio=0.8):
        """
        Splits a dictionary into train and test sets based on a specified ratio.
    
        Parameters:
        -curr_obs (dict): The input dictionary with keys as object IDs and values as observations (e.g., lists of log PDFs).
        -train_ratio (float): The ratio of the data to include in the training set (e.g., 0.8 for 80% train and 20% test).
    
        Returns:
        - train_dict: The training set dictionary.
        - test_dict: The test set dictionary.
        """
        TRAIN_RATIO=train_ratio
        keys = list(curr_obs.keys())
        random.shuffle(keys)

        # Calculate split index
        split_index = int(len(keys) * train_ratio)

        # Split keys and sort them
        train_keys = sorted(keys[:split_index])
        test_keys = sorted(keys[split_index:])

        # Create sorted train and test dictionaries
        train_dict = {key: curr_obs[key] for key in train_keys}
        test_dict = {key: curr_obs[key] for key in test_keys}

        return train_dict,test_dict
    