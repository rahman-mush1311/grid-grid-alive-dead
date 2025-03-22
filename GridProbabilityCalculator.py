import numpy 
import scipy.stats 
import math 

#String literals to constants
TRUE_LABELS = "true_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'

class GridProbabilityCalculator:
    def __init__(self, grid_rows=5, grid_cols=5, max_x=4128, max_y=2196):
        
        # self.n represents the number of observations for each cell
        self.n = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.mu represents mu_x,mu_y for each cell
        self.mu = [[(0, 0) for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.cov_mat represents the covariance for the each cell
        #to do 2*2 matrix initiliaziation
        self.cov_matrix = [[numpy.zeros((2, 2)) for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        self.max_x = max_x
        self.max_y = max_y
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # TODO: add statistics for normalizing standard deviation
           
    def compute_probabilities(self, observations,dx_norm, dy_norm, sx_norm, sy_norm):
        '''
        this calculates the probability of all objects using the given dead/alive model the calculation happens in probability(). for sanity checking purpose we don't consider if any object has only one set of coordinates.
        Parameters:
        -observations: dictionary containing object's observations(object id: (frame,x_cordinate,y_coordinate))
        -dx_norm: float value of mean of x_cordinates
        -dy_norm: float value of mean of y_coordinates
        -sx_norm: float value of variance of x_cordinates
        -sy_norm: float value of variance of y_coordinates
        
        Returns:
        -probabilities: a dictionary containing {object_id: {LOG_PDFS:list of log of probabilities}
        -minimum_probs: a list containing minimum of object's log probabilities(this is kept for thresholding purpose but might not need it)
        '''
        probabilities={}
        minimum_probs=[]
        empty_obs=0
        
        for obj_id, obs in observations.items():
            obj_probabilities=[]
            for i in range(len(obs) - 1):
                x,y=obs[i][1],obs[i][2]
                dframe = obs[i+1][0] - obs[i][0]
                dx = obs[i+1][1] - obs[i][1]
                dy = obs[i+1][2] - obs[i][2]
                if dframe>0:
                    dx,dy=(dx/dframe),(dy/dframe)
                    norm_dx = (dx - dx_norm) / sx_norm 
                    norm_dy = (dy - dy_norm) / sy_norm
                    probs=self.probability(x, y, norm_dx, norm_dy)    
                    obj_probabilities.append(probs)
                
                else:
                    print(f"not valid frame distance {dframe}")
            assert len(obj_probabilities) == len(obs)-1, f"Mismatch: {obj_id} has {len(obj_probabilities)} probabilities but {len(obs)-1} observations!"
            if len(obs)-1==0:
                empty_obs+=1
            
            log_obj_probabilities=self.log_probability(obj_probabilities)
            
            if len(log_obj_probabilities)>=1:
                probabilities[obj_id]={LOG_PDFS:log_obj_probabilities}
                minimum_probs.append(min(log_obj_probabilities))
                
        print(f"emptys are: {empty_obs}")  
        
        return probabilities,minimum_probs    
    def log_probability(self, curr_pdf_list):
        '''
        takes the list of log probabilities applies log transformations to that
        Parameters:
        -cuur_pdf_list: a list containing one object's probabilities
        Returns:
        log_values: a list of log applied probabilities
        '''
        log_values = [math.log(x) for x in curr_pdf_list if x != 0] 
        
        return log_values
        
    def probability(self, x, y, dx, dy):
        '''
        this calculates the probability of one objects using the particular cell's mu & covariance. we use the find_grid_cell() for that
        Parameters:
        -x: int value of x_cordinate 
        -y: int value of y_coordinates
        -dx: normalized displacement of x_cordinate
        -dy: normalized displacement of y_coordinates
        
        Returns:
        -curr_probability: float containing the calculated probabilities
        '''
        grid_row, grid_col = self.find_grid_cell(x, y)
        cell_mu = self.mu[grid_row][grid_col]
        cell_cov_matrix = self.cov_matrix[grid_row][grid_col]
        n = self.n[grid_row][grid_col]
        
        if n>=1:
            
            # TODO: create a 2-dimensional Gaussian distribution and use it to calculate a probability for (dx, dy)
            mvn = scipy.stats.multivariate_normal(mean=cell_mu , cov=cell_cov_matrix) #to do use the library name
            curr_probability=mvn.pdf((dx,dy))
            
            #print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} covariance is: {cell_cov_matrix}  probabilities: {curr_probability} for {dx,dy}")
            return curr_probability
        else:
            print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} sigma is: {cell_cov_matrix}  probabilities: empty for {dx,dy}")
            return 
            
    def combine_dictionary_observation(self, curr_obs, total_obs):
        '''
        this functions combines multiple dictionary observations.
        Parameters:
        -curr_obs: dictionary containing one dataset's object's observations(object id: (frame,x_cordinate,y_coordinate))
        -total_obs: dictionary containing multiple datasets object's observations(object id: (frame,x_cordinate,y_coordinate))
        Returns:
        -total_obs: modified with the curr_obs
        '''
        for obj_id,points in curr_obs.items():
            if obj_id in total_obs:
                continue
            else:
                # Add new object
                total_obs[obj_id] = points
                
        return total_obs
        
    def combine_data_with_labels(self,curr_log_pdf_dict,obs_dict_with_labels,label):
        '''
        this functions combines multiple dictionaries probabilities with labels.
        Parameters:
        -curr_log_pdf_dict: dictionary containing one dataset's object's probabilities{object_id: {LOG_PDFS:list of log of probabilities}}
        -obs_dict_with_labels: dictionary containing multiple datasets object's probabilities{object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}}
        -label: DEAD/ALIVE
        Returns:
        -obs_dict_with_labels: modified with the curr_log_pdf_dict
        '''
        for obj_id, values in curr_log_pdf_dict.items():
            if obj_id in obs_dict_with_labels:
                # Merge log values if object already exists
                obs_dict_with_labels.data[obj_id][LOG_PDFS].extend(values[LOG_PDFS])
                obs_dict_with_labels[obj_id][TRUE_LABELS]=label
            else:
                # Add new object
                obs_dict_with_labels[obj_id] = values
                obs_dict_with_labels[obj_id][TRUE_LABELS]=label
                
        return obs_dict_with_labels
        
    def find_grid_cell(self, x, y):
        grid_row = y * self.num_rows() // self.max_y
        grid_col = x * self.num_cols() // self.max_x
        return grid_row, grid_col

    def num_rows(self):
        return len(self.n)

    def num_cols(self):
        return len(self.n[0])