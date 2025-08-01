import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import one_hot_encoding

@dataclass
class DataTransformConfig:
    parsed_data_file_path=os.path.join('artifact',"parsed")
    final_data_file_path=os.path.join('artifact',"final")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformConfig()

    def get_data_transform_object(self)->dict:
        try:
            logging.info("Started getting data for transfromation")
            parsed_files_list = os.listdir(self.data_transformation_config.parsed_data_file_path)
            data_dict = {}

            for file in parsed_files_list:
                var_name = file.replace(".npy", "")
                path = os.path.join(self.data_transformation_config.parsed_data_file_path, file)
                data = np.load(path)
                data_dict[var_name] = data
                
            logging.info("Completed fetching data for transfromation")
            return data_dict
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, data_dict: dict) :
        try:
            logging.info("Started data transfromation")
            transformed_dict = {}

            for key, value in data_dict.items():
                if "x" in key.lower():
                    # Flatten all dimensions except the first
                    value = value.reshape(value.shape[0], -1)
                else:
                    value=one_hot_encoding(value)
                transformed_dict[key] = value

            os.makedirs(self.data_transformation_config.final_data_file_path,exist_ok=True)
            for key, value in transformed_dict.items():
                np.save(f"{self.data_transformation_config.final_data_file_path}/{key}.npy",value)
            logging.info("transfromation of data completed")
        except Exception as e:
            raise CustomException(e,sys)
            
        

            

        


        
            
