import numpy as np


def clip_grad_norm(layers, max_norm, norm_type=2.0):
    """
    Clips gradient norm of all parameters in layers.
    
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    
    Parameters
    ----------
    layers : list
        List of layer objects with gradients (dW, db, etc.)
    max_norm : float
        Maximum norm of the gradients.
    norm_type : float, optional
        Type of the used p-norm. Can be 'inf' for infinity norm.
        Default is 2.0 (L2 norm).
        
    Returns
    -------
    float
        Total norm of the gradients (viewed as a single vector).
        
    Examples
    --------
    >>> total_norm = clip_grad_norm(model.layers, max_norm=1.0)
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    
    grads = []
    for layer in layers:
        if hasattr(layer, 'dW') and layer.dW is not None:
            grads.append(layer.dW.flatten())
        if hasattr(layer, 'db') and layer.db is not None:
            grads.append(layer.db.flatten())
        if hasattr(layer, 'dgamma') and layer.dgamma is not None:
            grads.append(layer.dgamma.flatten())
        if hasattr(layer, 'dbeta') and layer.dbeta is not None:
            grads.append(layer.dbeta.flatten())
        if hasattr(layer, 'dWi') and layer.dWi is not None:
            grads.append(layer.dWi.flatten())
        if hasattr(layer, 'dWf') and layer.dWf is not None:
            grads.append(layer.dWf.flatten())
        if hasattr(layer, 'dWc') and layer.dWc is not None:
            grads.append(layer.dWc.flatten())
        if hasattr(layer, 'dWo') and layer.dWo is not None:
            grads.append(layer.dWo.flatten())
        if hasattr(layer, 'dWz') and layer.dWz is not None:
            grads.append(layer.dWz.flatten())
        if hasattr(layer, 'dWr') and layer.dWr is not None:
            grads.append(layer.dWr.flatten())
        if hasattr(layer, 'dWh') and layer.dWh is not None:
            grads.append(layer.dWh.flatten())
        if hasattr(layer, 'dUi') and layer.dUi is not None:
            grads.append(layer.dUi.flatten())
        if hasattr(layer, 'dUf') and layer.dUf is not None:
            grads.append(layer.dUf.flatten())
        if hasattr(layer, 'dUc') and layer.dUc is not None:
            grads.append(layer.dUc.flatten())
        if hasattr(layer, 'dUo') and layer.dUo is not None:
            grads.append(layer.dUo.flatten())
        if hasattr(layer, 'dUz') and layer.dUz is not None:
            grads.append(layer.dUz.flatten())
        if hasattr(layer, 'dUr') and layer.dUr is not None:
            grads.append(layer.dUr.flatten())
        if hasattr(layer, 'dUh') and layer.dUh is not None:
            grads.append(layer.dUh.flatten())
        if hasattr(layer, 'dbi') and layer.dbi is not None:
            grads.append(layer.dbi.flatten())
        if hasattr(layer, 'dbf') and layer.dbf is not None:
            grads.append(layer.dbf.flatten())
        if hasattr(layer, 'dbc') and layer.dbc is not None:
            grads.append(layer.dbc.flatten())
        if hasattr(layer, 'dbo') and layer.dbo is not None:
            grads.append(layer.dbo.flatten())
        if hasattr(layer, 'dbz') and layer.dbz is not None:
            grads.append(layer.dbz.flatten())
        if hasattr(layer, 'dbr') and layer.dbr is not None:
            grads.append(layer.dbr.flatten())
        if hasattr(layer, 'dbh') and layer.dbh is not None:
            grads.append(layer.dbh.flatten())
        if hasattr(layer, 'dE') and layer.dE is not None:
            grads.append(layer.dE.flatten())
    
    if len(grads) == 0:
        return 0.0
    
    all_grads = np.concatenate(grads)
    
    if norm_type == float('inf'):
        total_norm = np.max(np.abs(all_grads))
    else:
        total_norm = np.linalg.norm(all_grads, ord=norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-8)
    
    if clip_coef < 1:
        for layer in layers:
            if hasattr(layer, 'dW') and layer.dW is not None:
                layer.dW *= clip_coef
            if hasattr(layer, 'db') and layer.db is not None:
                layer.db *= clip_coef
            if hasattr(layer, 'dgamma') and layer.dgamma is not None:
                layer.dgamma *= clip_coef
            if hasattr(layer, 'dbeta') and layer.dbeta is not None:
                layer.dbeta *= clip_coef
            if hasattr(layer, 'dWi') and layer.dWi is not None:
                layer.dWi *= clip_coef
            if hasattr(layer, 'dWf') and layer.dWf is not None:
                layer.dWf *= clip_coef
            if hasattr(layer, 'dWc') and layer.dWc is not None:
                layer.dWc *= clip_coef
            if hasattr(layer, 'dWo') and layer.dWo is not None:
                layer.dWo *= clip_coef
            if hasattr(layer, 'dWz') and layer.dWz is not None:
                layer.dWz *= clip_coef
            if hasattr(layer, 'dWr') and layer.dWr is not None:
                layer.dWr *= clip_coef
            if hasattr(layer, 'dWh') and layer.dWh is not None:
                layer.dWh *= clip_coef
            if hasattr(layer, 'dUi') and layer.dUi is not None:
                layer.dUi *= clip_coef
            if hasattr(layer, 'dUf') and layer.dUf is not None:
                layer.dUf *= clip_coef
            if hasattr(layer, 'dUc') and layer.dUc is not None:
                layer.dUc *= clip_coef
            if hasattr(layer, 'dUo') and layer.dUo is not None:
                layer.dUo *= clip_coef
            if hasattr(layer, 'dUz') and layer.dUz is not None:
                layer.dUz *= clip_coef
            if hasattr(layer, 'dUr') and layer.dUr is not None:
                layer.dUr *= clip_coef
            if hasattr(layer, 'dUh') and layer.dUh is not None:
                layer.dUh *= clip_coef
            if hasattr(layer, 'dbi') and layer.dbi is not None:
                layer.dbi *= clip_coef
            if hasattr(layer, 'dbf') and layer.dbf is not None:
                layer.dbf *= clip_coef
            if hasattr(layer, 'dbc') and layer.dbc is not None:
                layer.dbc *= clip_coef
            if hasattr(layer, 'dbo') and layer.dbo is not None:
                layer.dbo *= clip_coef
            if hasattr(layer, 'dbz') and layer.dbz is not None:
                layer.dbz *= clip_coef
            if hasattr(layer, 'dbr') and layer.dbr is not None:
                layer.dbr *= clip_coef
            if hasattr(layer, 'dbh') and layer.dbh is not None:
                layer.dbh *= clip_coef
            if hasattr(layer, 'dE') and layer.dE is not None:
                layer.dE *= clip_coef
    
    return total_norm


def clip_grad_value(layers, clip_value):
    """
    Clips gradient values of all parameters in layers.
    
    Gradients are clipped element-wise to be within [-clip_value, clip_value].
    Gradients are modified in-place.
    
    Parameters
    ----------
    layers : list
        List of layer objects with gradients (dW, db, etc.)
    clip_value : float
        Maximum absolute value for gradients.
        
    Examples
    --------
    >>> clip_grad_value(model.layers, clip_value=0.5)
    """
    if clip_value <= 0:
        raise ValueError(f"clip_value must be positive, got {clip_value}")
    
    for layer in layers:
        if hasattr(layer, 'dW') and layer.dW is not None:
            np.clip(layer.dW, -clip_value, clip_value, out=layer.dW)
        if hasattr(layer, 'db') and layer.db is not None:
            np.clip(layer.db, -clip_value, clip_value, out=layer.db)
        if hasattr(layer, 'dgamma') and layer.dgamma is not None:
            np.clip(layer.dgamma, -clip_value, clip_value, out=layer.dgamma)
        if hasattr(layer, 'dbeta') and layer.dbeta is not None:
            np.clip(layer.dbeta, -clip_value, clip_value, out=layer.dbeta)
        if hasattr(layer, 'dWi') and layer.dWi is not None:
            np.clip(layer.dWi, -clip_value, clip_value, out=layer.dWi)
        if hasattr(layer, 'dWf') and layer.dWf is not None:
            np.clip(layer.dWf, -clip_value, clip_value, out=layer.dWf)
        if hasattr(layer, 'dWc') and layer.dWc is not None:
            np.clip(layer.dWc, -clip_value, clip_value, out=layer.dWc)
        if hasattr(layer, 'dWo') and layer.dWo is not None:
            np.clip(layer.dWo, -clip_value, clip_value, out=layer.dWo)
        if hasattr(layer, 'dWz') and layer.dWz is not None:
            np.clip(layer.dWz, -clip_value, clip_value, out=layer.dWz)
        if hasattr(layer, 'dWr') and layer.dWr is not None:
            np.clip(layer.dWr, -clip_value, clip_value, out=layer.dWr)
        if hasattr(layer, 'dWh') and layer.dWh is not None:
            np.clip(layer.dWh, -clip_value, clip_value, out=layer.dWh)
        if hasattr(layer, 'dUi') and layer.dUi is not None:
            np.clip(layer.dUi, -clip_value, clip_value, out=layer.dUi)
        if hasattr(layer, 'dUf') and layer.dUf is not None:
            np.clip(layer.dUf, -clip_value, clip_value, out=layer.dUf)
        if hasattr(layer, 'dUc') and layer.dUc is not None:
            np.clip(layer.dUc, -clip_value, clip_value, out=layer.dUc)
        if hasattr(layer, 'dUo') and layer.dUo is not None:
            np.clip(layer.dUo, -clip_value, clip_value, out=layer.dUo)
        if hasattr(layer, 'dUz') and layer.dUz is not None:
            np.clip(layer.dUz, -clip_value, clip_value, out=layer.dUz)
        if hasattr(layer, 'dUr') and layer.dUr is not None:
            np.clip(layer.dUr, -clip_value, clip_value, out=layer.dUr)
        if hasattr(layer, 'dUh') and layer.dUh is not None:
            np.clip(layer.dUh, -clip_value, clip_value, out=layer.dUh)
        if hasattr(layer, 'dbi') and layer.dbi is not None:
            np.clip(layer.dbi, -clip_value, clip_value, out=layer.dbi)
        if hasattr(layer, 'dbf') and layer.dbf is not None:
            np.clip(layer.dbf, -clip_value, clip_value, out=layer.dbf)
        if hasattr(layer, 'dbc') and layer.dbc is not None:
            np.clip(layer.dbc, -clip_value, clip_value, out=layer.dbc)
        if hasattr(layer, 'dbo') and layer.dbo is not None:
            np.clip(layer.dbo, -clip_value, clip_value, out=layer.dbo)
        if hasattr(layer, 'dbz') and layer.dbz is not None:
            np.clip(layer.dbz, -clip_value, clip_value, out=layer.dbz)
        if hasattr(layer, 'dbr') and layer.dbr is not None:
            np.clip(layer.dbr, -clip_value, clip_value, out=layer.dbr)
        if hasattr(layer, 'dbh') and layer.dbh is not None:
            np.clip(layer.dbh, -clip_value, clip_value, out=layer.dbh)
        if hasattr(layer, 'dE') and layer.dE is not None:
            np.clip(layer.dE, -clip_value, clip_value, out=layer.dE)
