#All necessary imports
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import tensorflow as tf
import argparse
import os
from tensorflow import keras

# https://stackoverflow.com/questions/64983112/keras-vertical-ensemble-model-with-condition-in-between
class G(tf.keras.Model):
    def __init__(self, B, B_prime):
        super(G, self).__init__()
        self.B = B
        self.B_prime = B_prime

    def predict(self, data):
        y = np.argmax(self.B(data), axis=1)
        y_prime = np.argmax(self.B_prime(data), axis=1)
        tmpRes = np.array([y[i] if y[i] == y_prime[i] else 1283 for i in range(y.shape[0])])
        res = np.zeros((y.shape[0], 1284))
        res[np.arange(tmpRes.size), tmpRes] = 1
        return res
    
    # For small amount of inputs that fit in one batch, directly using call() is recommended for faster execution,
    # e.g., model(x), or model(x, training=False) is faster then model.predict(x) and do not result in
    # memory leaks (see more details https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)

    def call(self, data):
        y = np.argmax(self.B(data), axis=1)
        y_prime = np.argmax(self.B_prime(data), axis=1)
        tmpRes = np.array([y[i] if y[i] == y_prime[i] else 1283 for i in range(y.shape[0])])
        res = np.zeros((y.shape[0], 1284))
        res[np.arange(tmpRes.size), tmpRes] = 1
        return res

# Load data
def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = np.transpose(x_data,(0,2,3,1))
    return x_data, y_data

# First create clones of the original badnet model (by providing the model filepath below)
# The result of repariting B_clone will be B_prime
def main():
    args = _parse_arguments()
    # Parameter Declaration
    clean_data_valid_filename = args.clean_valid
    clean_data_test_filename = args.clean_test
    poisoned_data_test_filename = args.bad_test

    global cl_x_valid, cl_y_valid, cl_x_test, cl_y_test, bd_x_test, bd_y_test
    cl_x_valid, cl_y_valid = data_loader(clean_data_valid_filename)
    cl_x_test, cl_y_test = data_loader(clean_data_test_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_test_filename)

    B = keras.models.load_model(args.bad_net)
    print("Begin testing on bad net..")
    test_robustness(B)

    
    # assign directory
    directory = args.save_path
    dir_list = [x for x in list(os.listdir(directory))]
    dir_list.sort()
    print(dir_list)
    # iterate over files in
    # that directory
    for filename in dir_list:

        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # print(f)
            B_prime = keras.models.load_model(f)
            # Repaired network reparied_net
            repaired_net = G(B, B_prime)
            print(f"Begin testing on \"{filename}\" pruned bad net..")

            test_robustness(repaired_net)




def test_robustness(model):
    
    cl_label_p = np.argmax(model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
    print(f'Clean classification accuracy: {clean_accuracy:.3f}%')

    bd_label_p = np.argmax(model.predict(bd_x_test), axis=1)
    success_rate = np.mean(np.equal(bd_label_p, bd_y_test))*100
    print(f'Attack success rate: {success_rate:.3f}%')

def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-cv", "--clean_valid", default="data/cl/valid.h5", type=str, 
        help="Path to clean, valid data")
    argparser.add_argument("-ct", "--clean_test", default="data/cl/test.h5", type=str, 
        help="Path to clean, test data")
    argparser.add_argument("-bt", "--bad_test", default="data/bd/bd_test.h5", type=str, 
        help="Path to bad, test data")
    argparser.add_argument("-bn", "--bad_net", default="models/bd_net.h5", type=str, 
        help="Path to model (_net)")
    argparser.add_argument("-sp", "--save_path", default="models/repaired", type=str, 
        help="Path to save pruned model")
    return argparser.parse_args()

if __name__ == "__main__":
    main()