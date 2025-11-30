import numpy as np


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    """
    
    def __init__(self, patience=10, min_delta=0, mode='min', restore_best_weights=True):
        """
        Initialize EarlyStopping callback.
        
        Args:
            patience: Number of epochs with no improvement to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'min' or 'max' - direction of improvement.
            restore_best_weights: Whether to restore model weights from best epoch.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def __call__(self, epoch, current, model=None):
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number.
            current: Current value of monitored metric.
            model: Model instance (for weight storage).
        
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best is None:
            self.best = current
            self.best_epoch = epoch
            if model and self.restore_best_weights:
                self._save_weights(model)
            return False
        
        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta
        
        if improved:
            self.best = current
            self.best_epoch = epoch
            self.counter = 0
            if model and self.restore_best_weights:
                self._save_weights(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            if model and self.restore_best_weights and self.best_weights:
                self._restore_weights(model)
            return True
        
        return False

    def _save_weights(self, model):
        """Save model weights."""
        self.best_weights = {}
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'W'):
                self.best_weights[f'W{i}'] = layer.W.copy()
                self.best_weights[f'b{i}'] = layer.b.copy()

    def _restore_weights(self, model):
        """Restore model weights from best epoch."""
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'W') and f'W{i}' in self.best_weights:
                layer.W = self.best_weights[f'W{i}']
                layer.b = self.best_weights[f'b{i}']


class ModelCheckpoint:
    """
    Save the model after every epoch or when monitored metric improves.
    """
    
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        """
        Initialize ModelCheckpoint callback.
        
        Args:
            filepath: Path to save the model weights.
            monitor: Metric name to monitor.
            mode: 'min' or 'max'.
            save_best_only: If True, only save when metric improves.
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None

    def __call__(self, epoch, logs, model):
        """
        Check if model should be saved.
        
        Args:
            epoch: Current epoch number.
            logs: Dictionary containing metric values.
            model: Model instance to save.
        """
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.best is None:
            self.best = current
            model.save(self.filepath)
            return
        
        if self.mode == 'min':
            improved = current < self.best
        else:
            improved = current > self.best
        
        if improved or not self.save_best_only:
            self.best = current
            model.save(self.filepath)


class History:
    """
    Record training history including losses and metrics.
    """
    
    def __init__(self):
        """Initialize History callback."""
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        """
        Record metrics at the end of an epoch.
        
        Args:
            epoch: Current epoch number.
            logs: Dictionary containing metric values.
        """
        logs = logs or {}
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def plot(self, metrics=None):
        """
        Plot training history.
        
        Args:
            metrics: List of metric names to plot. If None, plot all.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting")
            return
        
        if metrics is None:
            metrics = list(self.history.keys())
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric in self.history:
                ax.plot(self.history[metric])
                ax.set_title(metric)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
        
        plt.tight_layout()
        plt.show()
