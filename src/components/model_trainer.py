import os
import sys
import numpy as np
import math
from dataclasses import dataclass

from src.exceptions import CustomException
from src.logger import logging
from src.utils import compute_loss,compute_accuracy


from src.model.init import initialize_parameters
from src.model.forward import forward_prop
from src.model.backward import back_prop
from src.model.update import update_parameters

import pickle
@dataclass
class ModelTrainerConfig:
    training_data_file_path=os.path.join('artifact',"final")
    final_weight_file_path=os.path.join("artifact","trained_model")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def load_training_data(self):
        try:
            final_data_files_list = os.listdir(self.model_trainer_config.training_data_file_path)
            final_data_dict = {}

            for file in final_data_files_list:
                var_name = file.replace(".npy", "")
                path = os.path.join(self.model_trainer_config.training_data_file_path, file)
                data = np.load(path)
                final_data_dict[var_name] = data
                
            return final_data_dict
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_training(self,data_dict):
        try:
            logging.info("Started Model Training")
            # Transpose training data (shape: features x samples)
            data_dict["x_train"] = data_dict["x_train"].T  # shape: (784, N)
            data_dict["y_train"] = data_dict["y_train"].T  # shape: (10, N)

            x_train= data_dict["x_train"]  # shape: (784, N)
            y_train = data_dict["y_train"]  # shape: (10, N)

            assert np.max(x_train) <= 1.0 and np.min(x_train) >= 0.0
            hidden_layers=[784,512,256,128,10]
            parameters=initialize_parameters(layers_dims=hidden_layers)

            
            num_iters=1000
            curr_iter=0
            initial_lr = 0.009
            decay_rate = 0.99
            while curr_iter<num_iters:
                learning_rate = initial_lr * (decay_rate ** (curr_iter // 100))  # every 100 steps
                
                a4,cache=forward_prop(x_train,parameters=parameters)

                loss=compute_loss(y_train,a4)
                assert not np.isnan(loss) and not np.isinf(loss), "Loss exploded!"

                grad=back_prop(x_train,y_train,cache,parameters)

                parameters=update_parameters(parameters,grad,learning_rate)

                if curr_iter% math.ceil(num_iters / 10) == 0:
                    acc=compute_accuracy(y_train,a4)
                    print(f"Iteration {curr_iter:4d}: Cost {loss:.4f} Accuracy {acc:.4f}")
                    logging.info(f"Iteration {curr_iter:4d}: Cost {loss:.4f} Accuracy {acc:.4f}")
                    
                curr_iter+=1
            logging.info("Completed Model Training")
            os.makedirs(self.model_trainer_config.final_weight_file_path,exist_ok=True)
            with open(f"{self.model_trainer_config.final_weight_file_path}/model.pkl",'wb') as f:
                pickle.dump(parameters, f)
            logging.info("Model Saved")


        except Exception as e:
            raise CustomException(e,sys)








































            # W1 = np.random.randn(128, 784) * 0.01  # for small random initialization
        # b1 = np.zeros((128, 1))
        # W2 = np.random.randn(10, 128) * 0.01
        # b2 = np.zeros((10, 1))
        # Xavier Initialization
        # W1 = np.random.randn(128, 784) * np.sqrt(1 / 784)
        # b1 = np.zeros((128, 1))
        # W2 = np.random.randn(10, 128) * np.sqrt(1 / 128)
        # b2 = np.zeros((10, 1))



        
        

        