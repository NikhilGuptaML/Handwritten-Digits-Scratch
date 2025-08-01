import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import numpy as np


def initialize_parameters(layers_dims: list[int]) -> dict:
    try:
        logging.info("Weights creation initialized")
        parameters = {}

        for l in range(1, len(layers_dims)):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1] ) * np.sqrt(2. / layers_dims[l-1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
            
        logging.info("Weights creation completed")
        return parameters
    
    except Exception as e:
        raise CustomException(e,sys)