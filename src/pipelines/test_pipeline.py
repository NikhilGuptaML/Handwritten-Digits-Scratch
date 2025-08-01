import pickle
import numpy as np
from src.model.forward import forward_prop


import numpy as np

class TestPipeline:
    def run_pipeline(self):
        with open("artifact/trained_model/model.pkl", "rb") as f:
            parameters = pickle.load(f)

        x_test = np.load("artifact/final/x_test.npy").T
        y_test = np.load("artifact/final/y_test.npy").T

        a4,_=forward_prop(x_test,parameters)
        preds = np.argmax(a4, axis=0)
        true = np.argmax(y_test, axis=0)
        acc = np.mean(preds == true)*100
        print(f"Test Accuracy: {acc:.4f}")




