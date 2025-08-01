import logging
import os
from datetime import datetime

#Naming the log file
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"

#Selecting the path for log file to be in a folder named 'Logs/'
log_folder_path=os.path.join(os.getcwd(),'Logs',LOG_FILE)


#Making the directory and also checking if it exist then to not make a new one
os.makedirs(log_folder_path,exist_ok=True)

#When logging runs it creates the file with the name below
LOG_FILE_PATH=os.path.join(log_folder_path,f"{LOG_FILE}.log")

#Changing the basic logging function
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]- %(levelname)s -%(lineno)d %(name)s - %(message)s",
    level=logging.INFO
)

if __name__=="__main__":
    logging.info("Logging has started")