# Handwritten Digit Classifier from Scratch

This project implements a handwritten digit classifier built **entirely from scratch using NumPy**, trained on the **MNIST dataset** . It includes a modular training pipeline, data ingestion, preprocessing, and a test setup.

## Project Highlights

- Built without TensorFlow or PyTorch
- Uses a 4-layer fully-connected neural network
- Implements **forward + backward propagation manually**
- **Leaky ReLU + Softmax** activation functions
- Clean architecture using `src/` with logging and exception handling
  

---
## Architecture

Input Layer   : 784 (28x28 pixels)
Hidden Layer1 : 512 neurons (Leaky ReLU)
Hidden Layer2 : 256 neurons (Leaky ReLU)
Hidden Layer3 : 128 neurons (Leaky ReLU)
Output Layer  : 10 neurons (Softmax)
## Accuracy

Train Accuracy: 90.7650
Test Accuracy: 91.4800

## Project Structure

Handwritten-Digits-Scratch/<br>
├── artifact/ # Intermediate data & model files (auto-created)<br>
│ ├── parsed/<br>
│ ├── final/<br>
│ └── trained_model/<br>
├── data/ # Raw MNIST .gz files<br>
├── src/<br>
│ ├── components/ # Data ingestion, transformation, training<br>
│ ├── model/ # Model logic: forward, backward, update, init<br>
│ ├── pipelines/ # Training pipeline,Testing pipeline<br>
│ ├── utils.py # Utility functions (e.g., one-hot encoding)<br>
│ ├── exceptions.py # Custom error wrapper<br>
│ └── logger.py # Logging setup<br>
├── main.py # Entry point: runs the training,test pipeline<br>
├── .gitignore<br>
├── requirements.txt<br>
└── README.md<br>


---

## Features Implemented

- [x] Manual forward and backward propagation
- [x] Weight updates using gradient descent with learning rate decay
- [x] Accuracy and cross-entropy cost tracking
- [x] Modular folder structure
- [x] Model saving using `.pkl`
- [x] Image preprocessor to handle user-drawn digits

---

## How to Run

### 1. Download MNIST `.gz` files  using `download_data.py`

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run Python Sricpt
python main.py

### Future Improvements
Add early stopping or batch training

Auto-tune hyperparameters

Build a Streamlit UI

Author
Nikhil Gupta
B.Tech AIML 

