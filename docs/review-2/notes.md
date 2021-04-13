### Formula
Feature size is the value for matrix (if the feature size is 5 then, the it is a 5x5 matrix)
- (no padding)  Feature size = ((Image size - Kernel size) / Stride) + 1
- (padding)     Feature size = ((Image size + 2 \* Padding size - Kernel size) / Stride)+1

## Activation Function
- Used in the hidden layers of a neural network. This allows the model to learn more complex functions than a network trained using a linear activation function.
- Activation functions introduce non-linearity to the model which allows it to learn complex functional mappings between the inputs and response variables.

## ReLU (Rectified Linear Unit)
- Commonly used in Hidden Layer
- ReLU function is a piecewise linear function that outputs the input directly if is positive i.e. > 0, otherwise, it will output zero.
- ReLU(x) = max(0,x)

## Softmax
- Commonly used in Output Layer for Multiclass Classification
- It returns a Probablity distrubution of target classes for a Classification problem
- Neuron with the maximum Probablity is the target class
- Without Softmax the values in the output layer will not make sense
- Formula = e^z[i] / sumof(e^z[j])range(j,K)

### Links
- [Sofmax Formula Explaination](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer)

