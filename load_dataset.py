import sys
sys.path.append('./dataset')
import pickle
import numpy as np

def load_dataset(normalize = True, flatten = True):
    # Load dataset from pkl file
    # normalize: normalize data to 0-1 range
    # flattern: make the 2D array to 1D array
    """
    label type: one-hot-label
    """
    file_name = './datasets/mnist.pkl' # the name of dataset

    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    
    # Normalize image
    if normalize:
        dataset['train_img'] = dataset['train_img'].astype(np.float32)
        dataset['train_img'] /= 255.0
        dataset['test_img'] = dataset['test_img'].astype(np.float32)
        dataset['test_img'] /= 255.0

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = datset[key].reshape(-1, 1, 28, 28)

    # change label type
    def alter_one_hot_label(x):
        sample_array = np.zeros((x.size, 10))
        for index, row in enumerate(sample_array):
            row[x[index]] = 1

        return sample_array
    
    # change labe lype to one hot
    dataset['train_label'] = alter_one_hot_label(dataset['train_label'])
    dataset['test_label'] = alter_one_hot_label(dataset['test_label'])

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


(x_train, y_train), (x_test, y_test) = load_dataset()
