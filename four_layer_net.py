import numpy as np
from layer import *

# Neural Net with three layers
class FourLayerNet:
    """ initilize layers
    
    input_len : the length of input array
    hidden_node_num: the num of nodes in each hidden layer
    output_node_num: the num of node in output = the num of label
    weight_std: std of weight ( will be used for generating Gaussian distribution )
    
    Wn: [n]th weight
    bn: [n]th bias
    
    """
    def __init__(self, input_len, hidden_node_num, output_node_num, weight_std):
        self.params = {}
  
        # Gausian Dietribution is used for  Weight array
        # 1st layer
        self.params['W1'] = weight_std * np.random.randn(input_len, hidden_node_num)
        self.params['b1'] = np.zeros(hidden_node_num) # TODO change this

        # 2nd layer
        self.params['W2'] = weight_std * np.random.randn(hidden_node_num, input_len)
        self.params['b2'] = np.zeros(input_len) # TODO chaange this

        # 3rd layer
        self.params['W3'] = weight_std * np.random.randn(input_len, output_node_num)
        self.params['b3'] = np.zeros(output_node_num) # TODO change this



        """
        generate leyaers:
        Input -> DotLayer -> SigmoidLayer -> DotLayer -> SigmoidLayer ->
        Dotlayer -> SoftmaxLayer -> CrossEntropy -> output
        """

        self.layers = {}
        # w/out last layer
        self.layers_names = ['DotLayer1', 'Sigmoid1', 'DotLayer2', 'Sigmoid2', 'DotLayer3']
        
        self.layers[self.layers_names[0]] = DotLayer(self.params['W1'], self.params['b1'])
        self.layers[self.layers_names[1]] = SigmoidLayer()
        self.layers[self.layers_names[2]] = DotLayer(self.params['W2'], self.params['b2'])
        self.layers[self.layers_names[3]] = SigmoidLayer()
        self.layers[self.layers_names[4]] = DotLayer(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxLayerWithLoss()


    # predict the result with weight and bias
    def predict(self, x):
        """ x: image data
        """
        for layer in self.layers_names:
            x = self.layers[layer].forward(x) # move forward
        return x # will go to softmax function

    def loss(self, pre_out, t_label):
        # calc loss w/ cross_entropy method
        # used for last layer
        y = self.predict(pre_out)
        return self.last_layer.forward(y, t_label)

    def accuracy(self, x, t_label):
        y = self.predict(x)
        if t_label.ndim != 1: t_label = np.argmax(t_label, axis = 1)
        y = np.argmax(y, axis = 1 )
        accuracy = (np.sum(y==t_label) / float(x.shape[0]))
        return accuracy
        # x.shape: the num of the last node
        # accuracy: count up the num of nodes which has same label with t_label

    def cacl_back_propagation(self, x, t_label):
        # do forward to get loss, which is used for start back propagation
        self.loss(x, t_label)

        # back propagation
        dout = 1 # start num
        dout = self.last_layer.back_propagation(dout)

        layers_names = self.layers_names
        layers_names.reverse() # upside down
        for lay_name in layers_names:
            dout = self.layers[lay_name].back_propagation(dout)
        # ?
        layers_names.reverse()

        grads={}
        grads['W1'] = self.layers['DotLayer1'].dW
        grads['b1'] = self.layers['DotLayer1'].db
        grads['W2'] = self.layers['DotLayer2'].dW
        grads['b2'] = self.layers['DotLayer2'].db
        grads['W3'] = self.layers['DotLayer3'].dW
        grads['b3'] = self.layers['DotLayer3'].db
        
        return grads

        



