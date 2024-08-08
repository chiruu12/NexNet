import numpy as np
class ReLu:
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)
        return self.output
        
    def backward(self, gradient_output):
        self.diffv = np.maximum(0, gradient_output)
        return self.diffv


class Sigmoid:
    def forward(self,input):
        self.output=1/(1+np.exp(-input))
        return self.output
        
    def backward(self,gradient_output):
        self.diffv=gradient_output*(1-self.output)*self.output
        return self.diffv
        

class Tanh:
    def forward(self,input):
        
        self.input=input
        self.output=np.tanh(self.input)
        return self.output
        
    def backward(self, gradient_output):
        self.diffv = gradient_output * (1.0 - np.power((self.output), 2))
        return self.diffv
        
        
class Softmax:
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        e_values = np.exp(input)
        self.output = e_values / np.sum(e_values)
        return self.output
        
        
    def backward(self, gradient_output: np.ndarray) -> np.ndarray:
        grad_input = np.zeros_like(gradient_output)
        for i, (single_output, single_grad_output) in enumerate(zip(self.output, gradient_output)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            grad_input[i] = np.dot(jacobian_matrix, single_grad_output)
        return grad_input