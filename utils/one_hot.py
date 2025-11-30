import numpy as np


class OneHotEncoder:
    """
    One-Hot Encoder for converting between class labels and one-hot vectors.
    """
    
    def __init__(self, num_classes):
        """
        Initialize the OneHotEncoder.
        
        Args:
            num_classes: Total number of classes.
        """
        self.num_classes = num_classes

    def encode(self, labels):
        """
        Convert integer labels to one-hot encoded format.
        
        Args:
            labels: 1-D array of integer class labels of shape (num_samples,).
        
        Returns:
            One-hot encoded array of shape (num_samples, num_classes).
        """
        result = np.zeros((len(labels), self.num_classes), dtype=np.float32)
        result[np.arange(len(labels)), labels] = 1
        return result

    def decode(self, one_hot_matrix):
        """
        Convert one-hot encoded matrix back to class labels.
        
        Args:
            one_hot_matrix: One-hot encoded array of shape (num_samples, num_classes).
        
        Returns:
            Array of class labels of shape (num_samples,).
        """
        return np.argmax(one_hot_matrix, axis=1)

    def convert_to_one_hot(self, vector):
        """
        Alias for encode method for backward compatibility.
        
        Args:
            vector: 1-D array of integer class labels.
        
        Returns:
            One-hot encoded array.
        """
        return self.encode(vector)

    def one_hot_to_label(self, one_hot_matrix):
        """
        Alias for decode method for backward compatibility.
        
        Args:
            one_hot_matrix: One-hot encoded array.
        
        Returns:
            Array of class labels.
        """
        return self.decode(one_hot_matrix)


one_hot = OneHotEncoder
