import numpy as np
from src.utils import linear_transformation,leaky_relu,softmax

def forward_prop(X: np.ndarray, parameters: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:

    z1=linear_transformation(parameters["W1"],parameters["b1"],X)
    a1 = leaky_relu(z1)
    z2=linear_transformation(parameters["W2"],parameters["b2"],a1)
    a2 = leaky_relu(z2)
    z3=linear_transformation(parameters["W3"],parameters["b3"],a2)
    a3 = leaky_relu(z3)
    z4=linear_transformation(parameters["W4"],parameters["b4"],a3)
    a4 = softmax(z4)
    cache = {
    "Z1": z1, "A1": a1,
    "Z2": z2, "A2": a2,
    "Z3": z3, "A3": a3,
    "Z4": z4, "A4": a4
    }
    return (a4,cache)