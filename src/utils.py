import numpy as np
import numpy as np



def one_hot_encoding(array: np.array) -> np.array: #type: ignore
    num_classes = np.max(array) + 1
    oh_list = []
    for obj in array:
        zero_arr = np.zeros(num_classes)
        zero_arr[obj] = 1
        oh_list.append(zero_arr)
    return np.array(oh_list)


def linear_transformation(w,b,X):
    return w @ X + b


def matmul(X,Y):
    return X @ Y 

def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1.0/(1.0+np.exp(-z))
    return g

def softmax(z):
    """
    z: numpy array of shape (n_classes,) or (m, n_classes) if batch
    returns: same shape, with probabilities summing to 1
    """
    z_max = np.max(z, axis=0, keepdims=True)  # for numerical stability
    exp_z = np.exp(z - z_max)
    sum_exp_z = np.sum(exp_z, axis=0, keepdims=True)
    softmax_output = exp_z / sum_exp_z
    return softmax_output


def compute_loss(y_true, y_pred):
    m = y_true.shape[1]
    y_pred = np.clip(y_pred, 1e-8, 1.0)  
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss

def relu(z):
    """
    Compute ReLU of z (supports scalars or NumPy arrays)
    """
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def compute_accuracy(y_true, y_pred):
    pred_classes = np.argmax(y_pred, axis=0)
    true_classes = np.argmax(y_true, axis=0)
    return np.mean(pred_classes == true_classes) * 100



