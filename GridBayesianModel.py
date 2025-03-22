import numpy 
import scipy.stats 
import math 

#String literals to constants
TRUE_LABELS = "true_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'
DEAD_PDFS='dead_log_sum_pdfs'
ALIVE_PDFS='alive_log_sum_pdfs'

class GridProbabilityCalculator:
    def __init__(self, grid_rows=5, grid_cols=5, max_x=4128, max_y=2196):
        
        # self.dead_n represents the number of observations for each cell using dead observations
        self.dead_n = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.dead_mu represents mu_x,mu_y for each cell calculated using dead observations
        self.dead_mu = [[(0, 0) for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.dead_cov_mat represents the covariance for the each cell calculated using alive observations
        self.dead_cov_matrix = [[numpy.zeros((2, 2)) for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # self.dead_n represents the number of observations for each cell using dead observations
        self.alive_n = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.dead_mu represents mu_x,mu_y for each cell calculated using dead observations
        self.alive_mu = [[(0, 0) for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.dead_cov_mat represents the covariance for the each cell calculated using alive observations
        self.alive_cov_matrix = [[numpy.zeros((2, 2)) for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        self.max_x = max_x
        self.max_y = max_y
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
    
     def compute_probabilities(self, observations,dx_norm, dy_norm, sx_norm, sy_norm):
            
        probabilities={}
        
        empty_obs=0
        
        for obj_id, obs in observations.items():
            dead_obj_probabilities=[]
            alive_obj_probabilities=[]
            
            for i in range(len(obs) - 1):
                x,y=obs[i][1],obs[i][2]
                dframe = obs[i+1][0] - obs[i][0]
                dx = obs[i+1][1] - obs[i][1]
                dy = obs[i+1][2] - obs[i][2]
                if dframe>0:
                    dx,dy=(dx/dframe),(dy/dframe)
                    norm_dx = (dx - dx_norm) / sx_norm 
                    norm_dy = (dy - dy_norm) / sy_norm
                    
                    dead_probs=self.probability(x, y, norm_dx, norm_dy,DEAD)    
                    dead_obj_probabilities.append(dead_probs)
                    alive_probs=self.probability(x, y, norm_dx, norm_dy,ALIVE)    
                    alive_obj_probabilities.append(alive_probs)
                
                else:
                    print(f"not valid frame distance {dframe}")
            assert len(dead_obj_probabilities) == len(obs)-1, f"Mismatch: {obj_id} has {len(dead_obj_probabilities)} dead probabilities but {len(obs)-1} observations!"
            assert len(alive_obj_probabilities) == len(obs)-1, f"Mismatch: {obj_id} has {len(alive_obj_probabilities)} dead probabilities but {len(obs)-1} observations!"
            
            if len(obs)-1==0:
                empty_obs+=1
            
            dead_log_obj_probabilities=self.log_probability(dead_obj_probabilities)
            alive_log_obj_probabilities=self.log_probability(alive_obj_probabilities)
            
            if len(dead_log_obj_probabilities)>=1:
                probabilities[obj_id]={DEAD_PDFS:dead_log_obj_probabilities}
            if len(alive_log_obj_probabilities)>=1:
                probabilities[obj_id]={ALIVE_PDFS:alive_log_obj_probabilities}
                
        print(f"emptys are: {empty_obs}")  
        
        return probabilities
        
    def log_probability(self, curr_pdf_list):
        #print(f"before math:{len(curr_pdf_list)}, {curr_pdf_list}")   
        log_values = [math.log(x) for x in curr_pdf_list if x != 0] 
        #print(f"after math: {len(curr_pdf_list)}, {len(log_values)}")
        log_sum_values = np.sum(log_values)
        return log_sum_values
        
    def probability(self, x, y, dx, dy,model_type):
    
        grid_row, grid_col = self.find_grid_cell(x, y)
        
        cell_dead_mu = self.dead_mu[grid_row][grid_col]
        cell_dead_cov_matrix = self.dead_cov_matrix[grid_row][grid_col]
        dead_n = self.dead_n[grid_row][grid_col]
        alive_n = self.alive_n[grid_row][grid_col]
        cell_alive_mu = self.alive_mu[grid_row][grid_col]
        cell_alive_cov_matrix = self.alive_cov_matrix[grid_row][grid_col]
        
        if model_type==DEAD:
            if dead_n>=1:
                mvn = scipy.stats.multivariate_normal(mean=cell_dead_mu , cov=cell_dead_cov_matrix)
                curr_probability=mvn.pdf((dx,dy))
            
                #print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} covariance is: {cell_cov_matrix}  probabilities: {curr_probability} for {dx,dy}")
                return curr_probability
            else:
                print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} sigma is: {cell_cov_matrix}  probabilities: empty for {dx,dy}")
                return
        else:
            if alive_n>=1:
                mvn = scipy.stats.multivariate_normal(mean=cell_alive_mu , cov=cell_alive_cov_matrix)
                curr_probability=mvn.pdf((dx,dy))
            
                #print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} covariance is: {cell_cov_matrix}  probabilities: {curr_probability} for {dx,dy}")
                return curr_probability
            else:
                print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} sigma is: {cell_cov_matrix}  probabilities: empty for {dx,dy}")
                return
    
    def find_grid_cell(self, x, y):
        grid_row = y * self.num_rows() // self.max_y
        grid_col = x * self.num_cols() // self.max_x
        return grid_row, grid_col

    def num_rows(self):
        return len(self.n)

    def num_cols(self):
        return len(self.n[0])