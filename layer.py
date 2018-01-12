import numpy as np
from activation_functions import softmax 

"""
dit_layer: *W + b
SoftmaxLayer: layer which applies softmax(x) to the val
SigmoidLayer: layer which applis sigmoid(x) to the val

"""

class SigmoidLayer:
    def __init__(self):
        self.output = None
    
    # Prediction
    def forward(self, x):
        """ 
        x: the former layers's vals
        """
        delta = 1e-7
        output = 1 / ( 1 + np.exp(-x + delta) )
        self.output = output

        return output

    # BP
    def back_propagation(self, d):
        # update d(gradient)
        d = d*(1.0 - self.output ) * self.output

        return d # update

class SoftmaxLayerWithLoss:
    def __init__(self):
        self.loss = None
        self.softmax_out = None # output by softmax
        self.t_label = None # label array

    
    def forward(self, x, t_label):
        """
        x: input data
        """
        self.t_label = t_label
        self.softmax_out = softmax(x)
        self.loss = _cross_entropy_calc(self.softmax_out, self.t_label)
        
        return self.loss

    def back_propagation(self, dout = 1):
        batch_size = self.t_label.shape[0]
        dx = (self.softmax_out - self.t_label) / batch_size
        
        return dx

class DotLayer:
    """
    *W + bをするlayer
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.pre_out = None
        self.dW = None # for backward_propagation
        self.db = None # for backward_propagation

    def forward(self, pre_out):
        # for predictioni
        self.pre_out = pre_out
        pro_out = np.dot(pre_out, self.W) + self.b

        return pro_out # update output

    def back_propagation(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.pre_out.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dx


class Relu:
    def __init__(self):
        self.cover_array  = None # vals which are larger than 0

    def forward(self, pre_out):
        """
        if the val is larger than 0, just use the value as it is
        """
        self.cover_array = ( pre_out <= 0 )
        output = pre_out.copy()
        output[self.cover_array] = 0 # insert 0 

        return output

    # BP
    def back_propagation(self, dout):
        """
            return gradient
        """

        dout[self.cover_array] = 0 # 逆伝播でもReluの微小勾配は0
        dx = dout

        return dx

####### function
def _cross_entropy_calc(last_node_val, t_label):
    """ 
    calculate error by cross entropy method
    """
    delta = 1e-7 # to avoid over flow
    batch_size = last_node_val.shape[0] 
    return -np.sum(t_label*np.log(last_node_val + delta))/ batch_size
