from sklearn.datasets import fetch_openml
import numpy as np

def get_mnist_data():
    # here you define where you get your data from 
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(np.uint8)
    
    # before returing, you could do a preliminar preprocessing of your data
    # remember that if possible is better to embbed normalization in your model
    return X, y
    