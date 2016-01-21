import random
import time
import numpy as np
import gzip 
import cPickle
import time

def data_load(file):
    """
    Abosolute file path is passed as paramter. Loads data from that file. Returns a tuple .
    Tuple consists of test data , training data and validation data . 
    
    Each tuple consists of numpy arrays. 
    First array  contains 784x1 numpy arrays which represents the pixel intensities of the image. The second contains integers 
    representing the correct  classification digit.
    """
    
    f = gzip.open(file, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    return test_data,training_data, validation_data


def data_transform(file):
    """
    First function called to transform images into required form .

    Tranform the data into a format which is more feasible for training.
    
    Returns tuple  containing test data,training data and validation data   

    """
    
    data = data_load(file)
    t_d, v_d, tt_d = data[0], data[1], data[2]

    print type(t_d[1][1]), len(tt_d[1])
    X_test = [np.reshape(x, (784,1)) for x in tt_d[0]]
    test_data = zip(X_test, tt_d[1])
    X_train = [np.reshape(x, (784,1)) for x in t_d[0]]
    result = np.zeros((10,1))
    result[y] = 1
    Y_train = [result for y in t_d[1]]
    train_data = zip(X_train, Y_train)
    X_val = [np.reshape(x, (784,1)) for x in v_d[0]]
   
    val_data = zip(X_val, v_d[1])
    
    return test_data,train_data, val_data 
    
    
    
class NeuralNetwork(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
    
    def feedforward(self, network_inp):
        """
        Returns the output of a feedfoward network when input network_inp is given
        """
        
        a = network_inp
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b) 
        return a


    def update_mini_batch(self, mini_batches, eta):
        
        
        """
        Backpropagation algorithm is used to update parameters.
        
        mini_batches: array of mini_batches
        eta         : Learning Rate
        """
        
        nabla_w  = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        xv = np.asarray([x.ravel() for (x,y) in mini_batches]).transpose()
        yv = np.asarray([y.ravel() for (x,y) in mini_batches]).transpose()
        
        delta_b, delta_w = self.Backpropogation(xv,yv)
        
        nabla_w = [nw + ndw for  nw, ndw in zip(nabla_w, delta_w)]
        nabla_b = [nb + ndb for  nb, ndb in zip(nabla_b, delta_b)]
        self.weights = [w-(eta/len(mini_batches))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases =  [b-(eta/len(mini_batches))*nb for b, nb in zip(self.biases, nabla_b)]   


    def Stochastic_Gradient_Descent(self, training_data, epochs, mbs, eta, test_data = None):
        """

        training_data:         training data to perform Stochastic Gradient Descent.
        epochs:               Number of epochs or full iterations over the dataset.
        mbs:                  Size of mini-batch used.
        eta:                  Learning Rate
        test_data:            If test data is present, the function tests the model over
                                test data, and returns the accuracy.
        
        """
        for x in xrange(epochs):
            
            mini_batches = []
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), mbs):
                mini_batches.append(np.array(training_data[i:i+mbs]))
                
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Epoch : ",x, "Accuracy: ", self.check_performance(test_data), "/", len(test_data))
            else:
                print("Epoch : ",x, " Completed")
        
                
    
            
    def Backpropogation(self,x,y):
        """
        Backpropogation Algorithm
        """
        
        nabla_w  = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
    
        delta = (self.cost_derivative(activations[-1], y))*(first_derivate_sigmoid(zs[-1]))
        delta_s = delta.sum(1).reshape(len(delta), 1)
    
        nabla_b[-1] = delta_s
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for i in xrange(2, self.num_layers):
            
            delta = (np.dot(self.weights[-i + 1].transpose(), delta))*first_derivate_sigmoid(zs[-i])
            delta_s = delta.sum(1).reshape(len(delta), 1)

            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
            nabla_b[-i] = delta_s
        
        return nabla_b, nabla_w

            
    def check_performance(self, test_data):
        """
        Checks the performance of the neural network on test data.
        returns accuracy.
        
        """
        
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]       
        return sum([int(x==y) for x,y in test_results])
    
    def cost_derivative(self, output_activations, y):

           return (output_activations-y)
           
def sigmoid(z):
    """Sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def first_derivate_sigmoid(z):
    """First Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
     
file_path = input("Enter absolute path for dataset")
data = data_transform(file_path)


test_data,trainning_data, validation_data  = data[0], data[1], data[2]

neural_net = NeuralNetwork([784, 30,  10])

a = time.time()
neural_net.Stochastic_Gradient_Descent(trainning_data, 30, 10, 3.0, test_data = test_data)
b = time.time()
print "Time taken in seconds is ", b-a

