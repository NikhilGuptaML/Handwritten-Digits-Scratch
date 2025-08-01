import sys

from src.exceptions import CustomException

def update_parameters(parameters: dict, gradients: dict, learning_rate: float) -> dict:
    try:
        parameters["W1"]=parameters["W1"]-gradients["dw1"]*learning_rate
        parameters["b1"]=parameters["b1"]-gradients["db1"]*learning_rate
        parameters["W2"]=parameters["W2"]-gradients["dw2"]*learning_rate
        parameters["b2"]=parameters["b2"]-gradients["db2"]*learning_rate
        parameters["W3"]=parameters["W3"]-gradients["dw3"]*learning_rate
        parameters["b3"]=parameters["b3"]-gradients["db3"]*learning_rate
        parameters["W4"]=parameters["W4"]-gradients["dw4"]*learning_rate
        parameters["b4"]=parameters["b4"]-gradients["db4"]*learning_rate

        return parameters
    except Exception as e:
        raise CustomException(e,sys)

