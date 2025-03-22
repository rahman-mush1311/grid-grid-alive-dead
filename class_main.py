from observation_parser import ParsingObservations
from mean_covariance_plot import get_displacements
from GridDisplacementModel import GridDisplacementModel
from GridProbabilityCalculator import GridProbabilityCalculator
from GridModelEval import OutlierModelEvaluation
import numpy

TRUE_LABELS = "true_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'

def dead_with_dead_params_obs():
    '''
    first create a fileloader object which contains the filelists/dataset list (dead )and we can control how many to take as well.
    then we split in train & test sets calculate needed the parameters using GridDisplacementModel() object class and train sets only.
    Returns:
    dead_models_params: a dictionary {filename(key): dead_grid_model}
    -dead_train_obs: a nested dictionary of 80% observations {filename(key): {object_id: [(frame_1,x1,y1)...(frame_n,xn,yn)]}}
    -dead_train_obs: a nested dictionary of 20% observations {filename(key): {object_id: [(frame_1,x1,y1)...(frame_n,xn,yn)]}}
    '''
    dead_file_loader = ParsingObservations()
    dead_file_loader.load_files_from_folder(DEAD,3)
    
    dead_models_params = {}
    dead_train_obs={}
    dead_test_obs={}
    
    for files in dead_file_loader.filelists:
        #print(f" filename is {[files]}:")
        #print(f"observation size is:{len(dead_observations)}")
        dead_observations=dead_file_loader.load_observations([files]) 
        
        train_dead_observation,test_dead_observation=dead_file_loader.prepare_train_test(dead_observations,train_ratio=0.8)
        dead_grid_displacement_model=GridDisplacementModel()
      
        train_dead_grid_displacements=dead_grid_displacement_model.calculate_displacements(train_dead_observation)
        
        dead_grid_displacement_model.calculate_parameters(train_dead_grid_displacements)
       
        dead_models_params[files] = dead_grid_displacement_model 
        dead_train_obs[files]=train_dead_observation
        dead_test_obs[files]=test_dead_observation
        
    return dead_models_params,dead_train_obs,dead_test_obs,dead_file_loader
    
def get_combined_model(curr_file_loader,curr_model_params):  
    '''
    in here we combine the models to calculate the probabilities. since each models have different mu, covarince in each cell we need to combine to get generalized model
    Parameters:
    -curr_file_loader: fileloader object where dead/alive filelists are there
    -curr_model_params: all the model's parameters(grid based mu, covariances)
    Returns:
    -combined_model: grid displacement object with the calculated mu, covariances
    '''
    #created for combining all the models calculated
    combined_model = GridDisplacementModel()
    
    # Track the models
    calculated_models = []
    for i, files in enumerate(curr_file_loader.filelists):
        get_file = files
        if get_file not in curr_model_params:
            print(f"Warning: {get_file} not found in current_models_params.")
            continue  # Skip this file if not found

        # Store the last two models
        calculated_models.append(curr_model_params[get_file])

        # When reaching the last file, combine the last two models
        if i == len(curr_file_loader.filelists) - 1:  # Last file
            print(len(calculated_models))
            combined_model = combined_model.add_models(*calculated_models)  # Unpack list
            print(f"{get_file} This is the last file! Combined Model mu: {combined_model.mu}")
        else:
            print("Not all models to combine.")
    
    return combined_model
    
def calculate_with_dead_models(curr_file_loader,dead_outlier_model,curr_set,curr_obs_params,LABEL):
    '''
    we start calculating the probability with the dead model
    Parameters:
    -curr_file_loader:fileloader object where dead/alive filelists are there
    -dead_outlier_model:grid displacement object with the calculated mu, covariances using dead displacements
    -curr_set: train/test set of dead/alive observations{object_id: [(frame_1,x1,y1)...(frame_n,xn,yn)]}}
    -curr_obs_params: GridDisplacementModel() objects.
    -LABEL: curr_sets label dead/alive
    '''
    
    curr_probability_set={}   
    curr_point_set={}
    
    
    for i, files in enumerate(curr_file_loader.filelists):
        get_file = files
        curr_obs=curr_set[get_file]
        if get_file not in curr_obs_params:
            print(f"Warning: {get_file} not found in curr_obs_params.")
            continue  # Skip this file if not found
        else:
            dx_norm,dy_norm=curr_obs_params[get_file].total_mu[0],curr_obs_params[get_file].total_mu[1]
            sx_norm, sy_norm = numpy.sqrt(numpy.diag(curr_obs_params[get_file].total_cov_matrix))
            print(f"normalization values: {dx_norm, dy_norm, sx_norm, sy_norm}")
        
            curr_probs,min_dead_probs=dead_outlier_model.compute_probabilities(curr_obs,dx_norm, dy_norm, sx_norm, sy_norm)
            print(len(curr_probs))
            curr_probability_set=dead_outlier_model.combine_data_with_labels(curr_probs,curr_probability_set,LABEL) 
            print(len(curr_probability_set))
            curr_point_set=dead_outlier_model.combine_dictionary_observation(curr_obs,curr_point_set)
    
    return curr_point_set,curr_probability_set

def alive_params_with_alive_obs():
    '''
    first create a fileloader object which contains the filelists/dataset list (alive)and we can control how many to take as well.
    then we split in train & test sets calculate needed the parameters using GridDisplacementModel() object class and train sets only.
    Returns:
    -alive_models_params: a dictionary {filename(key): dead_grid_model}
    -alive_train_obs: a nested dictionary of 80% observations {filename(key): {object_id: [(frame_1,x1,y1)...(frame_n,xn,yn)]}}
    -alive_train_obs: a nested dictionary of 20% observations {filename(key): {object_id: [(frame_1,x1,y1)...(frame_n,xn,yn)]}}
    '''
    alive_file_loader = ParsingObservations()
    alive_file_loader.load_files_from_folder(ALIVE,2)
    
    alive_models_params = {}
    alive_train_observation={}
    alive_test_observation={}
    
    for files in alive_file_loader.filelists:
        print(f" filename is {[files]}:")
        alive_observations=alive_file_loader.load_observations([files]) 
        print(f"observation size is:{len(alive_observations)}")
        
        train_alive_observation,test_alive_observation=alive_file_loader.prepare_train_test(alive_observations,train_ratio=0.8)
        alive_grid_displacement_model=GridDisplacementModel()
        #print(f"Before: {alive_grid_displacement_model.mu}")
        train_alive_grid_displacements=alive_grid_displacement_model.calculate_displacements(train_alive_observation)
        alive_grid_displacement_model.calculate_parameters(train_alive_grid_displacements)
        
        dx_norm,dy_norm=alive_grid_displacement_model.total_mu[0],alive_grid_displacement_model.total_mu[1]
        sx_norm, sy_norm = numpy.sqrt(numpy.diag(alive_grid_displacement_model.total_cov_matrix))
        #print(f"normalization values: {dx_norm, dy_norm, sx_norm, sy_norm}")
        #print(f"After: {alive_grid_displacement_model.mu}")
        alive_models_params[files] = alive_grid_displacement_model
        alive_train_observation[files]=train_alive_observation
        alive_test_observation[files]=test_alive_observation
    
    return alive_file_loader,alive_train_observation,alive_test_observation,alive_models_params
    
def evaluate_train_model_with_threshold(curr_set):
    '''
    start performing the thresholds using the train set and prediction.
    Parameters:
    -curr_set: dictionary with object probabilities{object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}}
    Returns:
    -curr_updated_set: a dictionary {object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}{PREDICTED_LABELS: d/a}}
    -dead_outlier_model_eval: OutlierModelEvaluation() object with the calculated thresholds
    '''
    dead_outlier_model_eval=OutlierModelEvaluation()
    dead_outlier_model_eval.run_thresholding(curr_set)
    print(f"thresholds: {dead_outlier_model_eval.best_accuracy_threshold, dead_outlier_model_eval.best_precision_threshold}")
    curr_updated_set=dead_outlier_model_eval.predict_probabilities_dictionary_update(curr_set)
    
    return curr_updated_set,dead_outlier_model_eval
    
def run_outlier_model():

    dead_models_params,dead_train_observation_set,dead_test_observation_set,dead_file_loader=dead_with_dead_params_obs()   
    combined_dead_with_dead_model=get_combined_model(dead_file_loader,dead_models_params)
    
    dead_outlier_model= GridProbabilityCalculator()    
    dead_outlier_model.n=combined_dead_with_dead_model.n
    dead_outlier_model.mu =combined_dead_with_dead_model.mu
    dead_outlier_model.cov_matrix =combined_dead_with_dead_model.cov_matrix
                   
    dead_train_observation_set,dead_train_probability_set=calculate_with_dead_models(dead_file_loader,dead_outlier_model,dead_train_observation_set,dead_models_params,DEAD)
    
    alive_file_loader,alive_train_observation,alive_test_observation,alive_params=alive_params_with_alive_obs()
    
    alive_train_observation_set,alive_train_probability_set=calculate_with_dead_models(alive_file_loader,dead_outlier_model,alive_train_observation,alive_params,ALIVE)
   
    
    #print(alive_train_probability_set)
    train_probability_set=dead_train_probability_set|alive_train_probability_set
    #print(len(train_set))
    updated_train_probability_set,dead_outlier_model_eval=evaluate_train_model_with_threshold(train_probability_set)
    alive_train_ids=dead_outlier_model_eval.find_the_needed_obj_id(updated_train_probability_set,ALIVE,ALIVE)
    
    #dead_outlier_model_eval.plot_misclassied_obj(alive_false_ids, alive_train_observation_set)
    dead_test_observation_set,dead_test_probability_set=calculate_with_dead_models(dead_file_loader,dead_outlier_model,dead_test_observation_set,dead_models_params,DEAD)
    
    alive_test_observation_set,alive_test_probability_set=calculate_with_dead_models(alive_file_loader,dead_outlier_model,alive_test_observation,alive_params,ALIVE)
   
    
    #print(alive_train_probability_set)
    test_probability_set=dead_test_probability_set|alive_test_probability_set
    updated_test_probability_set=dead_outlier_model_eval.predict_probabilities_dictionary_update(test_probability_set)
    
    #dead_outlier_model_eval=OutlierModelEvaluation()
    alive_test_ids=dead_outlier_model_eval.find_the_needed_obj_id(updated_test_probability_set,ALIVE,ALIVE)

def run_bayesian_model():

    dead_models_params,dead_train_observation_set,dead_test_observation_set,dead_file_loader=dead_with_dead_params_obs()   
    combined_dead_with_dead_model=get_combined_model(dead_file_loader,dead_models_params)
    
    dead_outlier_model= GridProbabilityCalculator()    
    dead_outlier_model.n=combined_dead_with_dead_model.n
    dead_outlier_model.mu =combined_dead_with_dead_model.mu
    dead_outlier_model.cov_matrix =combined_dead_with_dead_model.cov_matrix
                   
    dead_train_observation_set,dead_train_probability_set=calculate_with_dead_models(dead_file_loader,dead_outlier_model,dead_train_observation_set,dead_models_params,DEAD)
    
    alive_file_loader,alive_train_observation,alive_test_observation,alive_params=alive_params_with_alive_obs()
    
    alive_train_observation_set,alive_train_probability_set=calculate_with_dead_models(alive_file_loader,dead_outlier_model,alive_train_observation,alive_params,ALIVE)
   
    
    #print(alive_train_probability_set)
    train_probability_set=dead_train_probability_set|alive_train_probability_set
    #print(len(train_set))
    updated_train_probability_set,dead_outlier_model_eval=evaluate_train_model_with_threshold(train_probability_set)
    alive_train_ids=dead_outlier_model_eval.find_the_misclassified_obj(updated_train_probability_set,ALIVE,ALIVE)
    
    #dead_outlier_model_eval.plot_misclassied_obj(alive_false_ids, alive_train_observation_set)
    dead_test_observation_set,dead_test_probability_set=calculate_with_dead_models(dead_file_loader,dead_outlier_model,dead_test_observation_set,dead_models_params,DEAD)
    
    alive_test_observation_set,alive_test_probability_set=calculate_with_dead_models(alive_file_loader,dead_outlier_model,alive_test_observation,alive_params,ALIVE)
   
    
    #print(alive_train_probability_set)
    test_probability_set=dead_test_probability_set|alive_test_probability_set
    updated_test_probability_set=dead_outlier_model_eval.predict_probabilities_dictionary_update(test_probability_set)
    
    #dead_outlier_model_eval=OutlierModelEvaluation()
    alive_test_ids=dead_outlier_model_eval.find_the_misclassified_obj(updated_test_probability_set,ALIVE,ALIVE)
    
if __name__ == "__main__":
   
    run_outlier_model()
    
    