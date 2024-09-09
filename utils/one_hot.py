import numpy as np
class one_hot:
    def __init__(self,num_classes):
        
        self.num_classes=num_classes
        
    def one_hot_to_label(self,one_hot_matrix):
        """
        Convert a one-hot encoded matrix to class labels.

        Args:
            y_one_hot : One-hot encoded array of shape (num_samples, num_classes).

        Returns:
            Array of class labels of shape (num_samples,).
        """
        return np.argmax(one_hot_matrix, axis=1)

    def convert_to_one_hot(self,vector):
        """
        Convert a vector of integer class labels to one-hot encoded format.

        Args:
            vector : 1-D array of integer class labels, shape (num_samples,).
            num_classes : Number of classes. If None, it is set to the maximum value in the vector + 1.

        Returns:
            2-D array of one-hot encoded labels, shape (num_samples, num_classes).
        """
        result = np.zeros((len(vector), self.num_classes), dtype=int)
        result[np.arange(len(vector)), vector] = 1
        return result
