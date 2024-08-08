import numpy as np

class CrossEntropyLoss:
    def forward(self, predictions, targets):
        
        exps = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)
        
        self.targets = targets
        n_samples = predictions.shape[0]
        self.loss = -np.sum(targets * np.log(np.clip(self.softmax, 1e-15, 1 - 1e-15))) / n_samples
        return self.loss

    def backward(self):
        grad_input = self.soft_max - self.targets
        return grad_input / self.targets.shape[0]
    
class MeanSquaredErrorLoss:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        self.loss = np.mean((predictions - targets) ** 2)
        return self.loss

    def backward(self):
        grad_input = 2 * (self.predictions - self.targets) / self.targets.size
        return grad_input