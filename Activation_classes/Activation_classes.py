import numpy as np
class ReLu:
    def forward(self, inputs):
        """
        Compute the forward pass of the ReLU activation function.

        Args:
            inputs (numpy.ndarray): The input array for the ReLU activation.

        Returns:
            numpy.ndarray: The output of the ReLU activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = inputs
        # Compute the ReLU activation: max(0, input)
        self.output = np.maximum(0, inputs)
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the ReLU activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the ReLU function
        # If the input was positive, the gradient is 1; otherwise, it is 0
        self.diffv = np.where(self.input > 0, gradient_output, 0)
        return self.diffv


class Sigmoid:
    def forward(self, input):
        """
        Compute the forward pass of the Sigmoid activation function.

        Args:
            input (numpy.ndarray): The input array for the Sigmoid activation.

        Returns:
            numpy.ndarray: The output of the Sigmoid activation (same shape as input).
        """
        # Compute the Sigmoid activation: 1 / (1 + exp(-input))
        self.output = 1 / (1 + np.exp(-input))
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Sigmoid activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the Sigmoid function
        # The gradient is: gradient_output * (1 - output) * output
        self.diffv = gradient_output * (1 - self.output) * self.output
        return self.diffv


class Tanh:
    def forward(self, input):
        """
        Compute the forward pass of the Tanh activation function.

        Args:
            input (numpy.ndarray): The input array for the Tanh activation.

        Returns:
            numpy.ndarray: The output of the Tanh activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = input
        # Compute the Tanh activation: np.tanh(input)
        self.output = np.tanh(self.input)
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Tanh activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the Tanh function
        # The gradient is: gradient_output * (1 - np.power(output, 2))
        self.diffv = gradient_output * (1.0 - np.power(self.output, 2))
        return self.diffv


class Softmax:
    def forward(self, inputs):
        """
        Perform the forward pass of the Softmax activation function.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Output data after applying the Softmax activation function, of the same shape as the input.
        """
        # Compute exponentiated values, with numerical stability
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize by the sum of all the exp values
        self.outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.outputs

    def backward(self, d_outputs):
        """
        Perform the backward pass of the Softmax activation function to compute gradients.

        Args:
            d_outputs (np.ndarray): Gradient of the loss with respect to the output of this layer, of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer, of the same shape as the input.
        """
        batch_size = self.outputs.shape[0]
        # Initialize gradient of the input
        d_inputs = np.zeros_like(d_outputs)

        # Compute gradients for each sample in the batch
        for i in range(batch_size):
            single_output = self.outputs[i].reshape(-1, 1)
            single_grad_output = d_outputs[i]

            # Jacobian matrix for the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Compute gradient for the current sample
            d_inputs[i] = np.dot(jacobian_matrix, single_grad_output)

        return d_inputs

class LeakyReLu:
    def __int__(self,alpha=0.01):
        """
        Initialize the LeakyReLU activation function with a learnable alpha parameter.

        Args:
            alpha_init (float): Initial value for the alpha parameter.
        """
        self.alpha=alpha
    def forward(self, inputs):
        """
        Compute the forward pass of the LeakyReLU activation function.

        Args:
            inputs (numpy.ndarray): The input array for the LeakyReLU activation.

        Returns:
            numpy.ndarray: The output of the LeakyReLU activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = inputs
        # Compute the LeakyReLU activation: max(small_constant*input, input)
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the ReLU activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the ReLU function
        # If the input was positive, the gradient is 1; otherwise, it is the aplha contstant 
        self.diffv = np.where(self.input > 0, gradient_output, self.alpha * gradient_output)
        return self.diffv

class ELU:
    def __init__(self,alpha=1) -> None:
        """
        Initialize the ELU activation function with a learnable alpha parameter.

        Args:
            alpha_init (float): Initial value for the alpha parameter.
        """
        self.alpha=alpha
    def forward(self, inputs):
        """
        Compute the forward pass of the ELU activation function.

        Args:
            inputs (numpy.ndarray): The input array for the ELU activation.

        Returns:
            numpy.ndarray: The output of the ELU activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = inputs
        # Compute the ELU activation: max(0, input)
        self.output = np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))
        return self.output
        
    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the ReLU activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the ReLU function
        # If the input was positive, the gradient is 1; otherwise, it is constant * exp(input)
        # genrally more computationally expensive 
        self.diffv = np.where(self.input > 0, gradient_output, gradient_output * self.alpha * np.exp(self.input))
        return self.diffv
    
class PReLU:
    def __init__(self, alpha_init=0.01):
        """
        Initialize the PReLU activation function with a learnable alpha parameter.

        Args:
            alpha_init (float): Initial value for the alpha parameter.
        """
        # Initialize alpha as a learnable parameter
        self.alpha = alpha_init
        self.alpha_grad = 0  # To store the gradient of alpha

    def forward(self, inputs):
        """
        Compute the forward pass of the PReLU activation function.

        Args:
            inputs (numpy.ndarray): The input array for the PReLU activation.

        Returns:
            numpy.ndarray: The output of the PReLU activation (same shape as input).
        """
        # Store the input for use in backward pass
        self.input = inputs
        # Compute the PReLU activation
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the PReLU activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradient of the PReLU function
        self.diffv = np.where(self.input > 0, gradient_output, self.alpha * gradient_output)
        
        # Compute the gradient with respect to alpha
        # For negative inputs,the gradient with respect to alpha is the sum of the gradient output
        self.alpha_grad = np.sum(np.where(self.input <= 0, gradient_output * self.input, 0))
        
        return self.diffv

    def update_alpha(self, learning_rate):
        """
        Update the alpha parameter using gradient descent.

        Args:
            learning_rate (float): The learning rate for the update.
        """
        self.alpha -= learning_rate*self.alpha_grad

class Swish:
    def __init__(self):
        """
        Initialize the Swish activation function.
        """
        self.sigmoid = Sigmoid()
        

    def forward(self, inputs):
        """
        Compute the forward pass of the Swish activation function.

        Args:
            inputs (numpy.ndarray): The input array for the Swish activation.

        Returns:
            numpy.ndarray: The output of the Swish activation (same shape as input).
        """
        self.inputs=inputs
        # Compute the sigmoid of the inputs using the Sigmoid class
        self.sigmoid_output = self.sigmoid.forward(inputs)
        # Compute the Swish activation
        self.output = inputs*self.sigmoid_output
        return self.output

    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Swish activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Get the gradient of the sigmoid output
        grad_sigmoid_output = self.sigmoid.backward(gradient_output)
        # Compute the gradient of the Swish function
        grad_input = gradient_output*(self.sigmoid_output + self.inputs*grad_sigmoid_output)
        return grad_input
    
class Softplus:
    def __init__(self):
        """
        Initialize the Softplus activation function.
        """
        self.sigmoid = Sigmoid()
        

    def forward(self, inputs):
        """
        Compute the forward pass of the Softplus activation function.

        Args:
            inputs (numpy.ndarray): The input array for the Softplus activation.

        Returns:
            numpy.ndarray: The output of the Softplus activation (same shape as input).
        """
        # Compute the Softplus activation
        self.input = inputs
        self.output = np.log(1 + np.exp(inputs))
        return self.output

    def backward(self, gradient_output):
        """
        Compute the backward pass (gradient) of the Softplus activation function.

        Args:
            gradient_output (numpy.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        # Compute the sigmoid of the input using the Sigmoid class
        sigmoid_input = self.sigmoid.forward(self.input)
        # The gradient of Softplus is the sigmoid of the input
        grad_input = gradient_output * sigmoid_input
        return grad_input