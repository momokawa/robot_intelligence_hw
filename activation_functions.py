# Activation Functions
import numpy as np

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )


def softmax(x):
    """ 

    normalize the values of result
    - the sum of result values is 1
    - the hightes value in array means highest possibility of prediction

    """
    exp_x = np.exp(x -np.max(x)) # to avoid overflow
    sum_exp = np.sum(exp_x)
    y = exp_x / sum_exp
    
    return y

