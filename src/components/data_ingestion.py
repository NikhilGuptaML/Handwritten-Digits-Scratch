import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import numpy as np
from dataclasses import dataclass
import gzip
import shutil
import struct

@dataclass
class DataIngestionConfig:
    raw_data_folder: str=os.path.join("data")
    updated_data_path: str=os.path.join('artifact','updated')
    parsed_data_path: str=os.path.join('artifact','parsed')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_unzipping_data(self):
        logging.info("Entered Data Ingestion")
        try:
            logging.info("Started unzipping the files")

            os.makedirs(os.path.dirname(self.ingestion_config.updated_data_path),exist_ok=True)

            raw_files_list=os.listdir(self.ingestion_config.raw_data_folder)

            for files in raw_files_list:
                name=files.replace(".gz","")
                with gzip.open(f"data/{files}", 'rb') as f_in:
                    with open(f"{self.ingestion_config.updated_data_path}/{name}", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            logging.info("Completed unzipping the files")
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_unpacking_data(self):
        try:
            logging.info("Started unpacking data")
            updated_file_list=os.listdir(self.ingestion_config.updated_data_path)
            os.makedirs(self.ingestion_config.parsed_data_path,exist_ok=True)
            for file in updated_file_list:
                if "images" in file:
                    
                    with open(f"{self.ingestion_config.updated_data_path}/{file}","rb") as f_read:

                        magic, num_images, rows, cols = struct.unpack('>IIII', f_read.read(16))

                        total_pixels=num_images*rows*cols
                        image_data=np.frombuffer(f_read.read(total_pixels),dtype=np.uint8)
                        processed_image_data=image_data.reshape(num_images,rows,cols)/255.0

                        if "train" in file :
                            new_file_name="x"+"_"+"train"
                        else:
                            new_file_name="x"+"_"+"test"

                        np.save(f"{self.ingestion_config.parsed_data_path}/{new_file_name}.npy",processed_image_data)
                        

                else:
                     with open(f"{self.ingestion_config.updated_data_path}/{file}","rb") as f_read:
                        magic, num_labels = struct.unpack('>II', f_read.read(8))
                        processed_label_data=np.frombuffer(f_read.read(num_labels),dtype=np.uint8)
                        
                        
                        if "train" in file :
                            new_file_name="y"+"_"+"train"
                        else:
                            new_file_name="y"+"_"+"test"

                        np.save(f"{self.ingestion_config.parsed_data_path}/{new_file_name}.npy",processed_label_data)
            logging.info("Completed unpacking the files")

        except Exception as e:
            raise CustomException(e,sys)
        

