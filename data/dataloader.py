import numpy as np


class DataLoader:
    """
    DataLoader for efficient batch processing of datasets.
    
    Provides an iterator interface for loading data in batches,
    with optional shuffling support for training.
    """
    
    def __init__(self, X, y, batch_size=32, shuffle=True):
        """
        Initialize the DataLoader.
        
        Args:
            X: Input features of shape (num_samples, ...).
            y: Target labels of shape (num_samples, ...).
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data at the start of each epoch.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / batch_size))
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        """
        Create an iterator for batching through the dataset.
        
        Yields:
            Tuple of (X_batch, y_batch) for each batch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        """
        Get the number of batches.
        
        Returns:
            Number of batches in the dataset.
        """
        return self.num_batches


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split arrays into random train and test subsets.
    
    Args:
        X: Input features of shape (num_samples, ...).
        y: Target labels of shape (num_samples, ...).
        test_size: Proportion of the dataset to include in the test split.
        shuffle: Whether to shuffle the data before splitting.
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(num_samples * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
