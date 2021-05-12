#!/usr/bin/env python
# coding: utf-8

# ### Importing

# In[104]:


import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# ### Iris dataset

# In[105]:


iris = pd.read_csv('iris.csv')


# In[106]:


class DataProcessing:
    @staticmethod
    def shuffle(Y):
        X = Y.copy()
        for i in range(len(X)-1, 0, -1):
            j = random.randint(0, i)
            X.iloc[i], X.iloc[j] = X.iloc[j], X.iloc[i]
        return X
    
    @staticmethod
    def splitSet(X):
        s = int(len(X)*0.7)
        train = X[:s]
        val = X[s:]
        return train, val
    
    @staticmethod
    def normalize(Y):
        X = Y.copy()
        for column in X.columns[:-1]: # [:-1] to exclude 'variety' column
            #min i max szukamy
            col_max = X[column].max()
            col_min = X[column].min()
            for row, x in enumerate(X[column]):
                if (col_max-col_min) != 0:
                    X.at[row, column] = (x-col_min) / (col_max-col_min)
        return X
    
    @staticmethod
    def one_hot_iris(y):
        # setosa = [1 0 0], versicolor = [0 1 0], virginica = [0 0 1]
        y_one_hot = []
        for i in range(len(y)):
            if y.iloc[i] == 'Setosa':
                y_one_hot.append([1, 0, 0])
            elif y.iloc[i] == 'Versicolor':
                y_one_hot.append([0, 1, 0])
            elif y.iloc[i] == 'Virginica':
                y_one_hot.append([0, 0, 1])
                
        return np.array(y_one_hot)
    
    @staticmethod
    def softmax_to_one_hot(vector):
        result = np.zeros(len(vector))
        max_value = max(vector)
        result[list(vector).index(max_value)] = 1
        return result
        


# In[107]:


irisNormalized = DataProcessing.normalize(iris)
irisMixed = DataProcessing.shuffle(irisNormalized)
irisMixedTrain, irisMixedVal = DataProcessing.splitSet(irisMixed)


# ### Classes Particle, PSO

# In[108]:


class Particle:
    def __init__(self, bounds):
        self.particle_velocity = [] # particle's velocity
        self.particle_position=[] # single particle vector with weights
        self.local_best_particle_position = [] # best local vector
        self.function_value = float("inf")
        self.local_best_particle_function_value = float("inf")
        
        for i in range(nv):
            self.particle_position.append(random.uniform(bounds[i][0], bounds[i][1]))
            self.particle_velocity.append(random.uniform(-1, 1))
    
    def __str__(self):
        return f"{self.particle_position} - {self.local_best_particle_position}" 
    
    def compute_function_values(self, net, X, y):
        # forward po sieci i liczenie bledu dla kazdego rekordu a wynikiem calej funkcji jest finalny blad sieci
        # computing mse for every sample and computing the final mse error
        errors = []
        probabilites = net.train(X)
        for i in range(len(probabilites)):
            error = mse(probabilites[i], y[i])
            errors.append(error)
        final_error = sum(errors) / len(X)
#         print(final_error)
        self.function_value = final_error
        
        # checking if new function value in smaller than the previous one
        if self.function_value < self.local_best_particle_function_value:
            self.local_best_particle_position = self.particle_position
            self.local_best_particle_function_value = self.function_value
    
    def move(self, phi1, phi2, global_best_particle_position, bounds): # moving particle
        # generating new particle's velocity
        for i in range(nv):
            w = random.random()
            c1 = random.random()
            c2 = random.random()
            
            self.particle_velocity[i] = w * self.particle_velocity[i] + phi1 * c1 * (self.local_best_particle_position[i] - self.particle_position[i]) + phi2 * c2 * (global_best_particle_position[i] - self.particle_position[i])
        
        # updating particle position
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

        # check and repair to satisfy the upper bounds
        if self.particle_position[i] > bounds[i][1]:
            self.particle_position[i] = bounds[i][1]
            
        # check and repair to satisfy the lower bounds
        if self.particle_position[i] < bounds[i][0]:
            self.particle_position[i] = bounds[i][0]
        
class PSO:
    def __init__(self):
        self.global_best_particle_position = []
        self.global_best_particle_function_value = float("inf") # best global vector
        self.particles = [] # all particles
        
    def run(self, T, n, phi1, phi2, bounds, X, y):  # train algortihm
        for i in range(n):
            self.particles.append(Particle(bounds))
            
        A = []
    
        for t in range(T):
            print(f'Epoch {t}..')
            for i in range(n):
                
                net = NeuralNetwork()
                
                input_neurons = 4
                hidden_layers_neurons = [8]
                output_neurons = 3
        
#                creating layers with weights from particles
                dense1 = DenseLayer(input_neurons, hidden_layers_neurons[0])
                dense1.weights = self.particles[i].particle_position[:input_neurons * hidden_layers_neurons[0]]
                dense1.weights = np.array(dense1.weights).reshape((hidden_layers_neurons[0], input_neurons))
#                 dense1.bias = np.array(self.particles[i].particle_position[21:24]).reshape((1, 3))
            
                dense2 = DenseLayer(hidden_layers_neurons[0], output_neurons)
                dense2.weights = self.particles[i].particle_position[input_neurons * hidden_layers_neurons[0]:input_neurons * hidden_layers_neurons[0] + 
                                                                    hidden_layers_neurons[0] * output_neurons]
                dense2.weights = np.array(dense2.weights).reshape((output_neurons, hidden_layers_neurons[0]))
#                 dense2.bias = np.array(self.particles[i].particle_position[24:]).reshape((1, 3))
            
                net.add_layer(dense1)
                net.add_layer(dense2)
                
                self.particles[i].compute_function_values(net, X, y) # neural network forward
                
                if self.particles[i].function_value < self.global_best_particle_function_value:
                    self.global_best_particle_position = list(self.particles[i].particle_position)
                    self.global_best_particle_function_value = float(self.particles[i].function_value)
            
            for i in range(n):
                self.particles[i].move(phi1, phi2, self.global_best_particle_position, bounds) # aktualizacja wag
                
            A.append(self.global_best_particle_function_value)
            
#         print('Optimal solution: ', self.global_best_particle_position)
        print('Objective function value: ', self.global_best_particle_function_value)
        return self.global_best_particle_position
    
def mse(X, y): # mean squred error for one sample
    result = 0
    for i in range(len(X)):
        result += (y[i] - X[i])**2
    return 0.5*result


# ## Neural Network

# In[109]:


class DenseLayer:
    def __init__(self, n_features, n_hidden_neurons):
        self.weights = np.random.rand(n_hidden_neurons, n_features)
#         self.weights = []
        self.bias = np.zeros((1, n_hidden_neurons))
    
    def set_weights(self, weights):
        for i in range(n_hidden_neurons):
            self.weights.append()
    
    def forward(self, inputs):
        self.output = np.dot(inputs, np.array(self.weights).T) + self.bias
        self.activation_tanh(self.output)
    
    def softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilites
    
    def activation_tanh(self, inputs):
        self.output = []
        for i in inputs:
            self.output.append(list(map(math.tanh, i)))
        self.output = np.array(self.output)

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def train(self, X):
        for k in range(len(self.layers)):
            self.layers[k].forward(X)
            X = self.layers[k].output


        output_layer = self.layers[-1]
        output_layer.activation_tanh(output_layer.output)
        
        output_softmax = output_layer.softmax(output_layer.output)
        return output_softmax   


# ### Parameters

# In[110]:


# bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  
bounds = [] # upper and lower bounds of variables
nv = 56  # number of variables (weights)
for i in range(nv):
    bounds.append((-1, 1))
    
# PARAMETERS
n = 1000  # number of particles
T = 300  # max number of iterations
# w = 0.75  # inertia constant
phi1 = 0.3  # cognative constant
phi2 = 0.8  # social constant


# In[111]:


X = irisMixedTrain.drop(columns=['variety'])
y = irisMixedTrain['variety']

X_train = X.to_numpy()
y_train = DataProcessing.one_hot_iris(y)


# In[112]:


start_timer_data = time.perf_counter()

pso = PSO()
best_weights = pso.run(T, n, phi1, phi2, bounds, X_train, y_train)  

end_timer_data = time.perf_counter()
print(f'Training takes: {end_timer_data - start_timer_data}s')


# In[113]:


X_val = irisMixedVal.drop(columns=['variety'])
y_val = irisMixedVal['variety']

X_val = X_val.to_numpy()
y_val = DataProcessing.one_hot_iris(y_val)


# In[114]:


input_neurons = 4
hidden_layers_neurons = [8]
output_neurons = 3

best_net = NeuralNetwork()

dense1 = DenseLayer(input_neurons, hidden_layers_neurons[0])
dense1.weights = best_weights[:input_neurons * hidden_layers_neurons[0]]
dense1.weights = np.array(dense1.weights).reshape((hidden_layers_neurons[0], input_neurons))

dense2 = DenseLayer(hidden_layers_neurons[0], output_neurons)
dense2.weights = best_weights[input_neurons * hidden_layers_neurons[0]:input_neurons * hidden_layers_neurons[0] + 
                                                                    hidden_layers_neurons[0] * output_neurons]
dense2.weights = np.array(dense2.weights).reshape((output_neurons, hidden_layers_neurons[0]))

best_net.add_layer(dense1)
best_net.add_layer(dense2)


softmax_output = best_net.train(X_val)
predictions = []
for i in range(len(softmax_output)):
    predictions.append(DataProcessing.softmax_to_one_hot(softmax_output[i]))

corrected = 0
for i in range(len(predictions)):
    comparison = y_val[i] == predictions[i]
    if comparison.all():
        corrected += 1

print(f'Accuracy = {(corrected / len(y_val))*100}%')


# ## Saving weights

# In[115]:


# import _pickle as cPickle

# with open('./weights_80.pickle', 'wb') as f:
#     cPickle.dump(best_weights, f)


# ## Loading best weights

# In[116]:


# with open(r"./weights_87.pickle", "rb") as f:
#     weights = cPickle.load(f)


# In[117]:


# loaded_net = NeuralNetwork()

# dense1 = DenseLayer(4, 3)
# dense1.weights = weights[:12]
# dense1.weights = np.array(dense1.weights).reshape((3, 4))

# dense2 = DenseLayer(3, 3)
# dense2.weights = weights[12:]
# dense2.weights = np.array(dense2.weights).reshape((3, 3))

# loaded_net.add_layer(dense1)
# loaded_net.add_layer(dense2)


# softmax_output = best_net.train(X_val)
# predictions = []
# for i in range(len(softmax_output)):
#     predictions.append(DataProcessing.softmax_to_one_hot(softmax_output[i]))

# corrected = 0
# for i in range(len(predictions)):
#     comparison = y_val[i] == predictions[i]
#     if comparison.all():
#         corrected += 1

# print(f'Accuracy = {(corrected / len(y_val))*100}%')

