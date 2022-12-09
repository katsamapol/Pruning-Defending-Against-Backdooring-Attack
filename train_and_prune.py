#All necessary imports
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import tensorflow as tf
import argparse
from tensorflow import keras
from keras import backend as K
from keras.models import Model

# Load data
def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = np.transpose(x_data,(0,2,3,1))
    return x_data, y_data

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

    # First create clones of the original badnet model (by providing the model filepath below)
    # The result of repariting B_clone will be B_prime
    B = keras.models.load_model(args.bad_net)

    B_clone = keras.models.load_model(args.bad_net)

    clean_accuracy = eval(B)
    _, _, _ = train_and_prune(clean_accuracy, B, B_clone, args.save_path)

def eval(B):
    # Get the original badnet model's (B) accuracy on the validation data
    cl_label_p = np.argmax(B(cl_x_valid), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_valid)) * 100

    print("Clean validation accuracy before pruning {0:3.6f}".format(clean_accuracy))
    K.clear_session()
    return clean_accuracy

def train_and_prune(clean_accuracy, B, B_clone, save_path):

    # layer to prune
    layer = 3

    # Redefine model to output right after the last pooling layer ("pool_3")
    intermediate_model=Model(inputs=B.inputs, outputs=B.get_layer('pool_'+str(layer)).output)

    # Get feature map for last pooling layer ("pool_3") using the clean validation data and intermidiate_model
    feature_maps_cl = intermediate_model(cl_x_valid)

    # Get average activation value of each channel in last pooling layer ("pool_3")
    averageActivationsCl = tf.math.reduce_mean(feature_maps_cl, axis=[0,1,2])

    # Store the indices of average activation values (averageActivationsCl) in increasing order
    # idxToPrune = tf.sort(averageActivationsCl, direction='ASCENDING') # increasing order
    idxToPrune = tf.argsort(averageActivationsCl, direction='ASCENDING') # increasing order 

    # Get the conv_4 layer weights and biases from the original network that will be used for prunning
    lastConvLayerWeights = B.get_layer('conv_'+str(layer)).get_weights()[0]
    lastConvLayerBiases = B.get_layer('conv_'+str(layer)).get_weights()[1]

    i=0
    save1 = False
    save2 = False
    pruned_arr = []
    clean_accuracy_valid_arr = []
    attack_success_rate_arr = []
    for chIdx in idxToPrune:
        i = i +1
        # Prune one channel at a time
        # Replace all values in channel 'chIdx' of lastConvLayerWeights and lastConvLayerBiases with 0
        lastConvLayerWeights[:,:,:,chIdx] = np.zeros_like(lastConvLayerWeights[:,:,:,chIdx])
        lastConvLayerBiases[chIdx]  = np.zeros_like(lastConvLayerBiases[chIdx])

        # Update weights and biases of B_clone
        B_clone.get_layer('conv_'+str(layer)).set_weights([lastConvLayerWeights, lastConvLayerBiases])


        # Evaluate the updated model's (B_clone) clean validation accuracy
        cl_label_p_valid = np.argmax(B_clone(cl_x_valid), axis=1)
        clean_accuracy_valid = np.mean(np.equal(cl_label_p_valid, cl_y_valid)) * 100
        
        # Evaluate the updated model's (B_clone) bad validation accuracy
        bd_label_p = np.argmax(B_clone(bd_x_test), axis=1)
        attack_success_rate = np.mean(np.equal(bd_label_p, bd_y_test))*100

        drop = abs(clean_accuracy_valid - clean_accuracy)/clean_accuracy
        
        # print(f"Round {i} of {len(idxToPrune)} drop: {drop*100:.2f}%, new clean: {clean_accuracy_valid:.2f}%, old clean: {clean_accuracy:.2f}%")
        pruned = i/len(idxToPrune)
        print(f"Round {i} of {len(idxToPrune)} pruned: {pruned*100:.2f}%, new clean accuracy: {clean_accuracy_valid:.2f}%, attack success rate: {attack_success_rate:.2f}%")
        pruned_arr.append(pruned)
        clean_accuracy_valid_arr.append(clean_accuracy_valid)
        attack_success_rate_arr.append(attack_success_rate)

        if (drop >= 0.02) and (save1 == False):
            # Save B_clone as B_prime and break
            print(f"Accuracy dropped by >=2%, current acc. is {clean_accuracy_valid:.2f}%")
            print("save model as \"g_net_00_02_percent.h5\"")
            B_clone.save(save_path+'g_net_00_02_percent.h5')
            save1 = True
        elif (drop >= 0.04) and (save2 == False) :
            # Save B_clone as B_prime and break
            print(f"Accuracy dropped by >=4%, current acc. is {clean_accuracy_valid:.2f}%")
            print("save model as \"g_net_00_04_percent.h5\"")
            B_clone.save(save_path+'g_net_00_04_percent.h5')
            save2 = True
        elif (drop >= 0.08):
            # Save B_clone as B_prime and break
            drop_text = str(f"{drop*100:4.2f}").replace(".","_")
            print(f"Accuracy dropped by >={drop*100:4.2f}%, current acc. is {clean_accuracy_valid:.2f}%")
            print(f"save model as \"g_net_{drop_text}_percent.h5\"")
            B_clone.save(save_path+f'g_net_{drop_text}_percent.h5')

    return pruned_arr, clean_accuracy_valid_arr, attack_success_rate_arr
        # If drop in clean_accuracy_valid is just greater (or equal to) than the desired threshold compared to clean_accuracy, then save B_clone as B_prime instead of B then break:
        # if (drop >= 0.02) and (save1 == False) :
        #     # Save B_clone as B_prime and break
        #     print(f"Accuracy dropped by >=2%, current acc. is {clean_accuracy_valid:.2f}%")
        #     print("save model as \"g_net_2_percent.h5\"")
        #     B_clone.save(save_path+'g_net_2_percent.h5')
        #     save1 = True
        # elif (drop >= 0.04) and (save2 == False) :
        #     # Save B_clone as B_prime and break
        #     print(f"Accuracy dropped by >=4%, current acc. is {clean_accuracy_valid:.2f}%")
        #     print("save model as \"g_net_4_percent.h5\"")
        #     B_clone.save(save_path+'g_net_4_percent.h5')
        #     save2 = True
        # elif (drop >= 0.8) and (save3 == False) :
        #     # Save B_clone as B_prime and break
        #     print(f"Accuracy dropped by >=8%, current acc. is {clean_accuracy_valid:.2f}%")
        #     print("save model as \"g_net_8_percent.h5\"")
        #     B_clone.save(save_path+'g_net_8_percent.h5')
        #     save3 = True
        # elif (drop >= 0.14) and (save4 == False) :
        #     # Save B_clone as B_prime and break
        #     print(f"Accuracy dropped by >=14%, current acc. is {clean_accuracy_valid:.2f}%")
        #     print("save model as \"g_net_14_percent.h5\"")
        #     B_clone.save(save_path+'g_net_20_percent.h5')
        #     save4 = True
        # elif (drop >= 0.22) and (save5 == False) :
        #     # Save B_clone as B_prime and break
        #     print(f"Accuracy dropped by >=40%, current acc. is {clean_accuracy_valid:.2f}%")
        #     print("save model as \"g_net_40_percent.h5\"")
        #     B_clone.save(save_path+'g_net_40_percent.h5')
        #     save5 = True
        # elif (drop >= 0.44) and (save6 == False) :
        #     # Save B_clone as B_prime and break
        #     print(f"Accuracy dropped by >=60%, current acc. is {clean_accuracy_valid:.2f}%")
        #     print("save model as \"g_net_60_percent.h5\"")
        #     B_clone.save(save_path+'g_net_60_percent.h5')
        #     save6 = True
        # elif (drop >= 0.72) and (save7 == False) :
        #     # Save B_clone as B_prime and break
        #     print(f"Accuracy dropped by >=80%, current acc. is {clean_accuracy_valid:.2f}%")
        #     print("save model as \"g_net_80_percent.h5\"")
        #     B_clone.save(save_path+'g_net_80_percent.h5')
        #     save7 = True
        #     print("Break")
        #     break
    

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
    argparser.add_argument("-sp", "--save_path", default="models/repaired/", type=str, 
        help="Path to save pruned model")
    return argparser.parse_args()

if __name__ == "__main__":
    main()