class SGD:
    def __init__(self, learning_rate):
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Perform a single optimization step by updating the weights and biases of the given layers.

        Args:
            layers (list of objects): List of layers in the network. Each layer should have attributes
                                      `W` (weights), `b` (biases), `dW` (gradient of weights), and `db` (gradient of biases).

        This method iterates over each layer and updates its weights and biases using the computed gradients.
        """
        for layer in layers:
            if hasattr(layer, 'W'):
                # Update weights and biases for layers with weight and bias attributes
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db

class one_hot:
    def __init__(self,num_classes):
        self.num_classes=num_classes
        
    
    def one_hot_to_label(self,one_hot_matrix):
        """
        Convert a one-hot encoded matrix to class labels.

        Args:
            y_one_hot (np.ndarray): One-hot encoded array of shape (num_samples, num_classes).

        Returns:
            np.ndarray: Array of class labels of shape (num_samples,).
        """
        return np.argmax(one_hot_matrix, axis=1)

    def convert_to_one_hot(self,vector):
        """
        Convert a vector of integer class labels to one-hot encoded format.

        Args:
            vector (np.ndarray): 1-D array of integer class labels, shape (num_samples,).
            num_classes (int, optional): Number of classes. If None, it is set to the maximum value in the vector + 1.

        Returns:
            np.ndarray: 2-D array of one-hot encoded labels, shape (num_samples, num_classes).
        """
        result = np.zeros((len(vector), self.num_classes), dtype=int)
        result[np.arange(len(vector)), vector] = 1
        return result
  