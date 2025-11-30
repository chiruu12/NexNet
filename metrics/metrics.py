import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true: Ground truth labels (one-hot or class indices).
        y_pred: Predicted labels (probabilities or class indices).
    
    Returns:
        Accuracy as a float between 0 and 1.
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average='macro'):
    """
    Calculate precision score.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'macro', 'micro', or 'weighted'.
    
    Returns:
        Precision score.
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0)
        weights.append(np.sum(y_true == cls))
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_pred == cls) & (y_true == cls)) for cls in classes])
        fp_total = np.sum([np.sum((y_pred == cls) & (y_true != cls)) for cls in classes])
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    elif average == 'weighted':
        return np.average(precisions, weights=weights)
    return precisions


def recall(y_true, y_pred, average='macro'):
    """
    Calculate recall score.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'macro', 'micro', or 'weighted'.
    
    Returns:
        Recall score.
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    weights = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0)
        weights.append(np.sum(y_true == cls))
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_pred == cls) & (y_true == cls)) for cls in classes])
        fn_total = np.sum([np.sum((y_pred != cls) & (y_true == cls)) for cls in classes])
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    elif average == 'weighted':
        return np.average(recalls, weights=weights)
    return recalls


def f1_score(y_true, y_pred, average='macro'):
    """
    Calculate F1 score.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'macro', 'micro', or 'weighted'.
    
    Returns:
        F1 score.
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    
    if p + r > 0:
        return 2 * (p * r) / (p + r)
    return 0


def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes (optional).
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return cm


def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
    
    Returns:
        MSE value.
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
    
    Returns:
        MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
    
    Returns:
        R² score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0
    return 1 - (ss_res / ss_tot)
