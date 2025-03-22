import re
import collections
import glob
import os 
import random

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
        print(self.filelists)            
    def load_observations(self,filenames):
        """
        Processes the input files and parses them to extract object ID, frame, x, and y coordinates.
        Parameters:
        -filenames: list of filenames to parse
        Returns:
        -observations: a dictionary (object id: (frame,x_cordinate,y_coordinate)).
        """
        pattern = re.compile(r'''
        \s*(?P<object_id>\d+),
        \s*(?P<within_frame_id>\d+),
        \s*'(?P<file_path>[^']+)',
        \s*cX\s*=\s*(?P<x>\d+),
        \s*cY\s*=\s*(?P<y>\d+),
        \s*Frame\s*=\s*(?P<frame>\d+)
        ''', re.VERBOSE)
        
        observations = collections.defaultdict(list)
        
        for filename in filenames:
            prefix, extension = self.get_file_prefix(filename)
            print(prefix,extension)
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
        
        for object_id, items in observations.items():
            assert all(items[i][0] <= items[i + 1][0] for i in range(len(items) - 1)), f"Items for {object_id} are not sorted by frame"
        
        
        return observations
    
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
            file_pattern = re.compile(r'''(\d{1,2}-\d{1,2}-\d{2})_(\d+)_ObjectXYs\.txt|AliveObjectXYs(\d+)([a-z])\.txt''')
            match = file_pattern.search(filename)
            if match.group(1)and match.group(2):
                return (match.group(1), match.group(2)) 
            else:
                return (match.group(3),match.group(4))
        return '',''
    
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
    