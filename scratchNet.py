import abc
from functools import total_ordering
from operator import index
import matplotlib.pyplot as plt
#from pkg_resources import _LegacyVersion
import tensorflow as tf
import numpy as np
import random
import time
from tqdm import tqdm, trange

# Minute 32:00 im Video 10.10.2022 11:44

def plot_loss_and_accuracy(losses, accuracies, xlabel):
    plt.plot(losses, label='loss')
    plt.plot(accuracies, label='accuracy')
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylim(top=1, bottom=0)
    plt.show()

class SquaredError:
    def derivative(self, net_input):
        raise NotImplementedError()
    
    def __call__(self, net_input):
        raise NotImplementedError() 

class DifferentialFuction(abc.ABC):
    '''Abstrakte Klasse einer differenzierbaren Funktion 
    Die Implementierung der Methoden erfolgt durch Spezialisierung'''
    
    def deriviate(self, net_input):
        pass
    
    def __call__(self, net_input):
        pass
        
        
class Sigmoid(DifferentialFuction):
    '''Sigmid Aktivierungsfunktion
    Stetige und differenzierbare Funktion, dessen Graph der Treppenfunktion ähnelt.'''        
    
    def deriviate(self, net_input):
        return self(net_input) * (1 - self(net_input))
        
    def __call__(self, net_input):
        return 1 / (1 + np.exp(-net_input))
    

class ScratchNet:
    
    def __init__(self, layers):
        self.learnin_rate = 0.5
        self.cost_func = SquaredError()
        self.layers = layers
        for index, layer in enumerate(self.layers):
            layer.prev_layer = self.layers[index - 1] if index > 0 else None
            layer.next_layer = (self.layers[index + 1] if index + 1 < len(self.layers) else None)
            layer.depth = index
            layer.initalize_parameters()
    
    def fit(self,train_images, train_labels, epochs=1):
        raise NotImplementedError()
    
    def predict(self, model_inputs):
        # Preprocessing der 'model inputs' durch den Input Layer
        model_inputs = self.layers[0].prepare_inputs(model_inputs)
        # TODO 
    def evaluate(self, validation_images, validation_labels):
        raise NotImplementedError()
    
    def compile(self, learning_rate=None, loss=None):
        raise NotImplementedError()
    
    def inspect(self):
        print(f"------- {self.__class__.__name__} ----------")
        print(f"   # Inputs: {self.layers[0].neuron_count}")
        for layer in self.layers:
            layer.inspect()
        
class DenseLayer:
    def __init__(
        self,
        neuron_count,
        depth=None,
        activation=None,
        biases=None,
        weights=None,
        prev_layer=None,
        next_layer=None):
        
        self.depth = depth
        self.next_layer = next_layer
        self.prev_layer = prev_layer
        
        self.neuron_count = neuron_count
        self.activation_func = activation or Sigmoid()
        
        self.weights = weights
        self.biases = biases
    
    def prepare_inputs(self, images, labels=None):
        return images if labels is None else images, labels
    
    def initalize_parameters(self):
        if self.weights is None:
            self.weights = np.random.randn(self.neuron_count, self.prev_layer.neuron_count)
        if self.biases is None:
            self.biases = np.random.randn(self.neuron_count, 1)    
            
    def inspect(self):
        print(f"------------- Layer L={self.depth} ----------")
        print(f"  # Neuronen: {self.neuron_count}")
        for n in range(self.neuron_count):
            print(f"       Neuron {n}")
            if self.prev_layer:
                for w in self.weights[n]:
                    print(f"       Weight: {w}")
                print(f"       Bias: {self.biases[n][0]}")            
                         
class FlattenLayer(DenseLayer):
    def __init__(self, input_shape):
        total_input_neurons = 1
        for dim in input_shape:
            total_input_neurons *= dim
        super().__init__(neuron_count=total_input_neurons)         
    
    def initalize_parameters(self):
        pass
    
    def prepare_inputs(self, images, labels=None):
        flattended_images = images.reshape(images.shape[0], self.neuron_count, 1)
        if labels is not None:
            labels = labels.reshape(labels.shape[0], -1, 1)
            return flattended_images, labels
        return flattended_images
        
def xor_keras():
    train_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_labels = np.array([[0], [1], [1], [0]])
    
    total_classes = 2
    train_vec_labels = tf.keras.utils.to_categorical(train_labels, total_classes)
    
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(2,1)),
            tf.keras.layers.Dense(4, activation="?"),
            tf.keras.layers.Dense(2, activation="?")
        ]
    )
    
    model.compile(optimizer="?", loss="?", metrics=["?"])
    
    model.fit(train_inputs, train_vec_labels, epochs=20)
    
    val_loss, val_accuracy = model.evaluate(train_inputs, train_vec_labels, verbose=False)
    print("Validation loss: %.2f" % val_loss)
    print("Validation accuracy: %.2f" % val_accuracy)
    
    
def xor():
    train_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_labels = np.array([[0], [1], [1], [0]])
    
    total_classes = 2
    train_vec_labels = tf.keras.utils.to_categorical(train_labels, total_classes)
    
    model = ScratchNet(
        [
            FlattenLayer(input_shape=(2,1)),
            DenseLayer(4, activation=Sigmoid()),
            DenseLayer(2, activation=Sigmoid())
        ]
    )
    #Wiederholt die Werte u weniger Epochen trainieren zu lassen
    repeat = (10000, 1)
    train_inputs = np.tile(train_inputs, repeat)
    train_vec_labels = np.tile(train_vec_labels, repeat)
    
    predicted = model.predict(np.array([[0, 0]]))
    print(predicted)
    return 
    
    model.compile(learning_rate=0.1, loss=SquaredError())
    
    start = time.time()
    losses, accuracies = model.fit(
        train_inputs, train_vec_labels, epochs=4)
    end = time.time()
    print("Trainingsdauer: {:.1f}s".format(end-start))
    
    model.fit(train_inputs, train_vec_labels, epochs=20)
    
    val_loss, val_accuracy = model.evaluate(train_inputs, train_vec_labels)
    print("Validation loss: %.2f" % val_loss)
    print("Validation accuracy: %.2f" % val_accuracy)    
    
    plot_loss_and_accuracy(losses, accuracies, xlabel="epochs")
    
xor()    