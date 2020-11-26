import numpy as np

e = np.exp(1)

np.random.seed(1)

synapticWeights = 2 * np.random.random((3, 1)) - 1

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


print("Random starting weights: ")
print(synapticWeights)


trainingInputs = np.array([[0.1,0.5,0.5],
                            [0.5,0.5,0.5],
                            [0.5,0.1,0.5],
                            [0.1,0.5,0.5]])

trainingOutputs = np.array([[0.1,0.5,0.5,0.1]]).T

  
train(trainingInputs, trainingOutputs, 1000)



#LET'S RUN IT!

print("Weights after training: ")
print(synapticWeights)
print("-------------------")


inputs = [0.5,0.1,0.5]
output = neuron(np.array(inputs))

#if output/0.5 < output/0.1:

    
#print("Input data = ", input1, input2, input3)
#print("-------------------")
print("Output data: ")
print(output)

