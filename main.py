# main
from load_dataset import load_dataset
from four_layer_net import *
import numpy as np
from layer import *
# from four_layer_net import *
from four_layer_net_Relu import * # Relu 使うとき


# read data set
(train_data, train_label), (test_data, test_label) = load_dataset()
 
n_net = FourLayerNet(input_len = 784, hidden_node_num = 50, output_node_num = 10, weight_std = 0.01)
# 28 x 28 = 784

iters_num = 10000 # TODO try to change here 

train_size =  train_data.shape[0]
batch_size = 200 # TODO try to change here
lr = 0.1 # learning rate

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size ,1 )

for i in range(iters_num):
    batch = np.random.choice(train_size, batch_size) # pick up batch samples
    x_batch = train_data[batch]
    t_batch = train_label[batch]
    
    # get gradient
    grads = n_net.cacl_back_propagation(x_batch, t_batch)

    # BP method update W and b
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        n_net.params[key] -= lr * grads[key] # update w and b
   
    loss = n_net.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # show the accuracies
    if i % iter_per_epoch == 0:
        train_acc = n_net.accuracy(train_data, train_label)
        test_acc = n_net.accuracy(test_data, test_label)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train accuracy:', train_acc, 'test accuracy:',  test_acc)

## def _show_plot(epoch_num, accuracy):



