import numpy 
import scipy.stats 
import math 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay

#String literals to constants
TRUE_LABELS = "true_labels"
PREDICTED_LABELS= "predicted_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'
DEAD_PDFS='dead_log_sum_pdfs'
ALIVE_PDFS='alive_log_sum_pdfs'

class BayesianModel:
    def __init__(self, grid_rows=5, grid_cols=5, max_x=4128, max_y=2196):
        
        self.prior_dead=0.0
        self.prior_alive=0.0
        
        self.filtered_thresholds = []
        self.best_accuracy_threshold = None
        self.best_precision_threshold = None
        self.best_classify = -float('inf')
        self.best_precision = -float('inf')
    
    def calculate_prior(self,dead_train_obs,alive_train_obs):
        """
        calculate the prior probabilities according to the number of points, the forumla is: #of alive or dead objects/total number of object's (according to trainning)
    
        Parameters:
        -alive_train_obs: The dictionary with alive keys as object IDs and values as observations (e.g., lists of log PDFs).
        -dead_train_obs: The dictionary with dead keys as object IDs and values as observations (e.g., lists of log PDFs).
    
        """
        self.prior_dead=len(dead_train_obs)/(len(dead_train_obs)+len(alive_train_obs))
        self.prior_alive=len(alive_train_obs)/(len(dead_train_obs)+len(alive_train_obs))
       
        return
    
    def sum_log_probabilities(self, curr_obs_with_dead, curr_obs_with_alive):
        
        curr_likelihood={}
        
        for obj_id in curr_obs_with_dead:
            cls=''
            valid_dead_log_pdfs = [v for v in curr_obs_with_dead[obj_id][LOG_PDFS] if v != 0]
            valid_alive_log_pdfs = [v for v in curr_obs_with_alive[obj_id][LOG_PDFS] if v != 0]        
            if not valid_dead_log_pdfs or not valid_alive_log_pdfs:
                print(f"Warning: Invalid log_pdfs for obj_id {obj_id}. {valid_dead_log_pdfs} {valid_alive_log_pdfs}")
                continue
       
            # Compute the log posterior probabilities
            dead_log_sum_pdf = numpy.sum(valid_dead_log_pdfs) + numpy.log(self.prior_dead)
            alive_log_sum_pdf = numpy.sum(valid_alive_log_pdfs) + numpy.log(self.prior_alive)
        
            #print(f"dead_log_sum is {dead_log_sum_pdf}, alive_log_sum_pdf: {alive_log_sum_pdf}")
            if dead_log_sum_pdf>alive_log_sum_pdf:
                cls=DEAD
            else:
                cls=ALIVE
                
            curr_likelihood[obj_id] = {
                DEAD_PDFS: dead_log_sum_pdf,
                ALIVE_PDFS: alive_log_sum_pdf,
                TRUE_LABELS: curr_obs_with_dead[obj_id][TRUE_LABELS],
                PREDICTED_LABELS: cls
            }          
        
        return curr_likelihood
        
    def plot_confusion_matrix(self, curr_obs, obs_type):
    
        true_label = [curr_obs[obj_id][TRUE_LABELS] for obj_id in curr_obs]
        predicted_label= [curr_obs[obj_id][PREDICTED_LABELS] for obj_id in curr_obs]
        print(len(true_label), len(predicted_label))
        
        # Create the confusion matrix
        cm = confusion_matrix(true_label, predicted_label, labels=[DEAD, ALIVE])
        accuracy = accuracy_score(true_label, predicted_label)
        f1 = f1_score(true_label, predicted_label, pos_label='a', average='binary')
        recall = recall_score(true_label, predicted_label, pos_label='a', average='binary')
        precision = precision_score(true_label, predicted_label, pos_label='a', average='binary')
        print(f"{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}")
        
         # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dead (0)", "Alive (1)"])
        disp.plot(cmap="Greens")
        disp.ax_.set_title(f"Confusion Matrix Using {obs_type}")
        disp.ax_.set_xlabel("Predicted Labels")
        disp.ax_.set_ylabel("True Labels")
    
        metrics_text = (
            f"Accuracy: {accuracy:.3f}\n"
            f"F1-Score: {f1:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"Precision: {precision:.3f}\n"
            
        )
        #f"Threshold: {self.best_accuracy_threshold:.3f}"
        disp.ax_.legend(
            handles=[
                plt.Line2D([], [], color='white', label=metrics_text)
            ],
            loc='lower right',
            fontsize=10,
            frameon=False
        )
    
        plt.show()
        
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