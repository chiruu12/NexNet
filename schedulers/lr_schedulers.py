import numpy as np


class StepLR:
    """
    Step learning rate scheduler.
    
    Decays the learning rate by a factor every specified number of epochs.
    """
    
    def __init__(self, optimizer, step_size, gamma=0.1):
        """
        Initialize the StepLR scheduler.
        
        Args:
            optimizer: Optimizer instance whose learning rate will be adjusted.
            step_size: Number of epochs between each decay.
            gamma: Multiplicative factor for learning rate decay.
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.learning_rate
        self.epoch = 0

    def step(self):
        """Update learning rate based on current epoch."""
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.learning_rate = self.base_lr * (self.gamma ** (self.epoch // self.step_size))


class ExponentialLR:
    """
    Exponential learning rate scheduler.
    
    Decays the learning rate by gamma every epoch.
    """
    
    def __init__(self, optimizer, gamma=0.95):
        """
        Initialize the ExponentialLR scheduler.
        
        Args:
            optimizer: Optimizer instance whose learning rate will be adjusted.
            gamma: Multiplicative factor for learning rate decay per epoch.
        """
        self.optimizer = optimizer
        self.gamma = gamma
        self.base_lr = optimizer.learning_rate
        self.epoch = 0

    def step(self):
        """Update learning rate based on current epoch."""
        self.epoch += 1
        self.optimizer.learning_rate = self.base_lr * (self.gamma ** self.epoch)


class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler.
    
    Adjusts learning rate using a cosine annealing schedule.
    """
    
    def __init__(self, optimizer, T_max, eta_min=0):
        """
        Initialize the CosineAnnealingLR scheduler.
        
        Args:
            optimizer: Optimizer instance whose learning rate will be adjusted.
            T_max: Maximum number of epochs.
            eta_min: Minimum learning rate.
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.learning_rate
        self.epoch = 0

    def step(self):
        """Update learning rate based on current epoch."""
        self.epoch += 1
        self.optimizer.learning_rate = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * self.epoch / self.T_max)) / 2


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    """
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6):
        """
        Initialize the ReduceLROnPlateau scheduler.
        
        Args:
            optimizer: Optimizer instance whose learning rate will be adjusted.
            mode: 'min' or 'max' - whether to minimize or maximize the metric.
            factor: Factor by which to reduce learning rate.
            patience: Number of epochs with no improvement after which LR will be reduced.
            min_lr: Minimum learning rate.
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metric):
        """
        Update learning rate based on metric value.
        
        Args:
            metric: Current value of the monitored metric.
        """
        if self.best is None:
            self.best = metric
            return
        
        if self.mode == 'min':
            improved = metric < self.best
        else:
            improved = metric > self.best
        
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.optimizer.learning_rate * self.factor, self.min_lr)
            self.optimizer.learning_rate = new_lr
            self.num_bad_epochs = 0
