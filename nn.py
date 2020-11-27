import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from data import trainingInputs, trainingOutputs

np.random.seed(1)

#                                       _ number of elements in input
synapticWeights = 2 * np.random.random((16, 1)) - 1

e = np.exp(1)
def softplus(x):
    return np.log(1+((e)**x))
 
def softplusDerivative(x):
    return 1/(1+((e)**-x))


def neuron(inputs):
    inputs = inputs.astype(float) 
    return softplus(np.dot(inputs, synapticWeights))

def train(trainingInputs, trainingOutputs, trainingIterations):
    global synapticWeights
    for iteration in range(trainingIterations):
        
        output = neuron(trainingInputs)
        
        error = trainingOutputs - output
        
        adjustments = np.dot(trainingInputs.T, error * softplusDerivative(output))
        
        synapticWeights += adjustments


def predict(inputs):
  output = neuron(np.array(inputs))
  y = np.array([abs(0.1 - output[0]), abs(0.2 - output[0]), abs(0.3 - output[0]), abs(0.4 - output[0])])
  key = np.array(["bottom-top (0.1)", "Top-bottom (0.2)", "Right Line(0.3)", "Left Line(0.4)"])


  
  print("Output Data: ", colored(( output), 'green'), "\n")

  sortedOut = y.argsort()
  print("    Differences: ", colored((y[sortedOut]), 'blue'))
  print("Ordered Outputs: ", colored((key[sortedOut]), 'blue'))
  #np.sort(y)

  smallest = y[0]
  index = 0
  for i in range(len(y)):
    if y[i] < smallest:
      smallest = min(smallest, y[i])
      index = i

  print("Smallest Difference: ", smallest, "\n")

  print("Prediction: ", colored((key[index]), 'green'), "\n")

def plot():
  plt.scatter([synapticWeights], inputs, alpha=0.5)
  plt.title('My Model')
  plt.xlabel('Weights')
  plt.ylabel('Inputs')
  plt.show()

def save():
  np.save("weights.npy", synapticWeights)

def load():
  global synapticWeights
  synapticWeights = np.load("weights.npy")





#RUN



#Data from data.py
train(trainingInputs, trainingOutputs, 100000)
#load()

save()

inputs = [0, 0, 0, 1, \
          0, 0, 0, 1, \
          0, 0, 0, 1, \
          0, 0, 0, 1]
predict(inputs)


plot()






























