import numpy as np
class Initializer:
    def __init__(self, method='xavier'):
        """
        Initialize the Initializer with the specified method for weights.

        Args:
            method : Initialization method for weights ('xavier', 'he', 'random', 'zero'). taken xavier as default as it is one 
            of the better initializers.
        """
        self.method = method

    def initialize_weights(self, input ,output):
        """
        Initialize weights based on the specified method.

        Args:
            input : number of inputs for the layer.
            outpur : number of outputs for the layer.

        Returns:
            Initialized weights for the layer
        """
        shape = (input, output)
        if self.method == 'xavier':
            # Xavier Initialization:
            # Useful for sigmoid, tanh, and softmax activation functions.
            # It helps maintain the variance of gradients across layers, avoiding vanishing/exploding gradients.
            # so it has been used as the default optimizer as well
            limit = np.sqrt(6 / (input + output))
            return np.random.uniform(-limit, limit, size=(input,output))

        elif self.method == 'he':
            # He Initialization:
            # Best suited for ReLU and its variants. also preserves variance 
            # and helps avoid issues with dying neurons in ReLU-based networks.
            stddev = np.sqrt(2. / input )
            return np.random.randn(*shape) * stddev

        elif self.method == 'random':
            # Random Initialization:
            # Simple initialization for small networks or models.
            # Not recommended for deep networks due to potential instability.
            return np.random.uniform(-0.1, 0.1, size=shape)

        elif self.method == 'zero':
            # Zero Initialization:
            # Not typically used for weights as it can cause symmetry issues.
            return np.zeros(shape)

        else:
            # as only 4 initializers have been added.
            raise ValueError(f"Unsupported initialization method: {self.method}")