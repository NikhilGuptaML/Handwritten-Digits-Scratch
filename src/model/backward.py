import numpy as np
from src.utils import matmul,leaky_relu_derivative

def back_prop(x_train,y_train,cache,parameters) -> dict[str, np.ndarray]:
    _, m = x_train.shape

    dz4 = cache["A4"] - y_train

    dw4 = (matmul(dz4 , cache["A3"].T)) / m
    db4 = np.sum(dz4, axis=1, keepdims=True) / m


    dz3 = (matmul(parameters["W4"].T , dz4)) * leaky_relu_derivative(cache["Z3"])
    dw3 = (matmul(dz3 , cache["A2"].T)) / m
    db3 = np.sum(dz3, axis=1, keepdims=True) / m

    dz2 = (matmul(parameters["W3"].T , dz3)) * leaky_relu_derivative(cache["Z2"])
    dw2 = (matmul(dz2 , cache["A1"].T)) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = (matmul(parameters["W2"].T , dz2)) * leaky_relu_derivative(cache["Z1"])
    dw1 = (matmul(dz1 , x_train.T)) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    gradients = {
    "dw1": dw1, "db1": db1,
    "dw2": dw2, "db2": db2,
    "dw3": dw3, "db3": db3,
    "dw4": dw4, "db4": db4
    }

    return gradients
    