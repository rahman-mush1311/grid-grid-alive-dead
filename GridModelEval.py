import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay

TRUE_LABELS = "true_labels"
PREDICTED_LABELS= "predicted_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'

class OutlierModelEvaluation:
    def __init__(self, window=2):
    
        self.window_size = window
        self.filtered_thresholds = []
        self.best_accuracy_threshold = None
        self.best_precision_threshold = None
        self.best_classify = -float('inf')
        self.best_precision = -float('inf')
    
    
    def run_thresholding(self,curr_obs):
        """
        this function selects thresholds from curr_obs dictionary using get_thresholds_from_roc(). Additionally, this function does the classification according to the thresholds got from roc_curve, it looks at different window to consider the classification
        the metrics to evaluate is not perfect
        Parameters:
        - curr_obs dictionary {object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}}
        Returns:
        - N/A
    
        """
        self.get_thresholds_from_roc(curr_obs)
        
        for threshold in self.filtered_thresholds:
            predictions = []
            true_labels = []
            
            for obj_id, values in curr_obs.items():
                cls = DEAD
                log_values = values[LOG_PDFS]
                for i in range(len(log_values) - self.window_size + 1):
                    w = log_values[i:i+self.window_size]
                    if all(p <= threshold for p in w):
                        cls = ALIVE
                predictions.append(1 if cls == DEAD else 0)
                true_labels.append(1 if values[TRUE_LABELS] == DEAD else 0)
            if len(set(true_labels)) < 2:
                print(f"Warning: Only one class present in true_labels: {set(true_labels)}. Skipping evaluation.")
                #continue
            
            cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions)
            classify = cm[0, 0] + cm[1, 1]  # True positives + True negatives
            
            if classify > self.best_classify:
                self.best_classify = classify
                self.best_accuracy_threshold = threshold
            if self.best_precision < precision:
                self.best_precision_threshold = threshold
                self.best_precision = precision
        
        print(f"Optimal Threshold: {self.best_accuracy_threshold}, Maximum Classification: {self.best_classify}, Best Precision: {self.best_precision}, Threshold for Best Precision: {self.best_precision_threshold}")
        return 
    
    def get_thresholds_from_roc(self,curr_obs):
        '''
        it filters furthers when the tpr improves fpr decreases, it rounds the fpr, tpr values to 2 digits get more relevant thresholds. assigns to the class's self.filtered_thresholds list
        '''
        true_labels=[]
        log_pdf_values=[]
        for obj_data in curr_obs.values():
            log_pdf_values.extend(obj_data[LOG_PDFS])
            true_labels.extend([1 if obj_data[TRUE_LABELS] == ALIVE else 0] * len(obj_data[LOG_PDFS]))
    
        true_labels=np.array(true_labels)
        log_pdf_values=np.array(log_pdf_values)
        print(true_labels.shape,log_pdf_values.shape)
        fpr, tpr, roc_thresholds = roc_curve(true_labels, log_pdf_values)
        #print(min(roc_thresholds),max(roc_thresholds))
    
        for i in range(1, len(roc_thresholds)):
            if round(tpr[i],2) > round(tpr[i - 1],2) or round(fpr[i],2)<round(fpr[i-1],2):
                self.filtered_thresholds.append(roc_thresholds[i])
        
        '''
        plt.figure(figsize=(10, 6))
        # Plot the ROC curve
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.show()
        '''
        return 
    
    def predict_probabilities_dictionary_update(self,curr_obs):
        '''
        this function final classification is done with best threshold which gives maximum classification. It prints the confusion matrices also.
        Parameters:
        - curr_obs :dictionary {object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}}
        Returns:
        - curr_obs :dictionary {object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}{PREDICTED_LABELS: d/a}}
        '''
        for obj_id in curr_obs:
            cls = DEAD
            for i in range(len(curr_obs[obj_id][LOG_PDFS]) - self.window_size + 1):
                w = curr_obs[obj_id][LOG_PDFS][i:i+self.window_size]
                if all([p <= self.best_accuracy_threshold for p in w]):
                    cls = ALIVE

        # Update the dictionary with predicted and true labels
            curr_obs[obj_id] = {
                LOG_PDFS: curr_obs[obj_id][LOG_PDFS],  # Original log PDF values
                TRUE_LABELS: curr_obs[obj_id][TRUE_LABELS],
                PREDICTED_LABELS: cls
            }
        
        true_labels = [curr_obs[obj_id][TRUE_LABELS] for obj_id in curr_obs]
        predicted_labels = [curr_obs[obj_id][PREDICTED_LABELS] for obj_id in curr_obs]

        # Create the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=[DEAD, ALIVE])
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, pos_label='a', average='binary')
        recall = recall_score(true_labels, predicted_labels, pos_label='a', average='binary')
        precision = precision_score(true_labels, predicted_labels, pos_label='a', average='binary')
        print(f"{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}")
        
         # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dead (0)", "Alive (1)"])
        disp.plot(cmap="Blues")
        disp.ax_.set_title(f"  Confusion Matrix Using ")
        disp.ax_.set_xlabel("Predicted Labels")
        disp.ax_.set_ylabel("True Labels")
    
        metrics_text = (
            f"Accuracy: {accuracy:.3f}\n"
            f"F1-Score: {f1:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"Precision: {precision:.3f}\n"
            f"Threshold: {self.best_accuracy_threshold:.3f}"
        )
        disp.ax_.legend(
            handles=[
                plt.Line2D([], [], color='white', label=metrics_text)
            ],
            loc='lower right',
            fontsize=10,
            frameon=False
        )
    
        plt.show()
        return curr_obs
    
    def find_the_needed_obj_id(self,curr_obs,label_true,label_predicted):
        '''
        we find the list of ids to the analysis.
        Parameters:
        -curr_obs:{object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}{PREDICTED_LABELS: d/a}}
        -label_true: DEAD/ALIVE
        -label_predicted: DEAD/ALIVE
        Returns:
        -ids: list of ids where the labels meet the given conditions
        '''
        ids = [ obj_id for obj_id, details in curr_obs.items()
        if details[TRUE_LABELS] == label_true and details[PREDICTED_LABELS] == label_predicted
        ]
        print(len(ids))
        print(ids)
        
        return ids
        
    def get_prefix(self,obj_id):
        '''
        parsers the object_id for the purpose of which dataset it belongs to. it needs more work
        Parameters:
        -obj_id: str containing obj_id
        Returns:
        -D/A/date
        '''
        if obj_id.startswith("D"):
            return "D"
        elif obj_id.endswith("a") or obj_id.endswith("p"):
            return "A"
        else:
            return "1-6-25" 
            
    def plot_extracted_obj(self, extracted_ids, observations,label_true,label_predicted):
        '''
        we analysis the object's trajectory. label_true and label_predicted could be matching or not matching depending upon our analysis.
        
        Parameters:
        
        -extracted_ids: list of ids
        -observations: dictionary containing object's observations(object id: (frame,x_cordinate,y_coordinate)) 
        -label_predicted: DEAD/ALIVE
        -label_predicted: DEAD/ALIVE
        Returns:
        -N/A
        '''
        prefix_colors = {
        "D": "red",
        "A": "blue",
        "1-6-25": "green"
        }
        i=0
        for obj_id in misclassified_ids:
            x=[]
            y=[]
            if obj_id in observations:
                points = observations[obj_id]
                x = [p[1] for p in points]  # Extract x-coordinates
                y = [p[2] for p in points]  # Extract y-coordinates
                prefix = self.get_prefix(obj_id)  # Determine prefix
                color = prefix_colors.get(prefix, "black")  # Assign color (default to black if unknown)
    
                plt.plot(x, y, marker="o", linestyle="-", color=color, label=prefix if prefix not in plt.gca().get_legend_handles_labels()[1] else "")  # Plot trajectory

                # Labels & Formatting
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.title(f"{label_true} Object Labels Predicted {label_predicted} Train Set")
                plt.legend(title=f"Object id {obj_id}")
                plt.grid(True, linestyle="--", alpha=0.6)
                #plt.savefig(f"ALIVE Object Labels Predicted DEAD Train Set {i}.png")
                i+=1
                # Show the plot
                plt.show()
        
        
      
    
   