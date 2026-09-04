"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    if seed is not None:
        np.random.seed(seed)
        
    indices = np.random.permutation(X.shape[0])
    return X[indices], y[indices]
    pass

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n_samples = X.shape[0]
    
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    
    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    pass

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    """
    Computes per-feature mean and standard deviation arrays along columns.
    Safely replaces standard deviations of 0 with 1 to prevent division-by-zero.
    Returns: (mean, std) as a tuple of NumPy arrays.
    """
    # Force the input array into float coordinates immediately
    X_numeric = np.asarray(X, dtype=float)
    
    # Calculate across columns (axis=0)
    mean = np.mean(X_numeric, axis=0)
    std = np.std(X_numeric, axis=0)
    
    # Replace any standard deviation of exactly 0.0 with 1.0
    std = np.where(std == 0.0, 1.0, std)
    
    # Return as a clean, unpackable tuple
    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    """
    Standardizes X using a precomputed stats tuple (mean, std).
    """
    X_numeric = np.asarray(X, dtype=float)
    return (X_numeric - mean) / std
    pass

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    """
    Prepends a vertical column of 1.0 values to the left of feature matrix X.
    """
    X_numeric = np.asarray(X, dtype=float)
    n_samples = X_numeric.shape[0]
    ones = np.ones((n_samples, 1))
    return np.hstack((ones, X_numeric))
    pass

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    """
    Combines standardization and bias insertion into a unified preprocessing engine.
    """
    X_scaled = standardize_features(X, mean, std)
    return add_bias_column(X_scaled)
    pass

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return np.asarray(X, dtype=float) @ np.asarray(weights, dtype=float)
    pass

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))
    pass

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    X_mat = np.asarray(X, dtype=float)
    n_samples = X_mat.shape[0]
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return (2.0 / n_samples) * (X_mat.T @ errors)
    pass

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    """
    Computes the exact, analytical global minimum vector using least-squares.
    """
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float)
    
    # np.linalg.lstsq returns a tuple; index 0 isolates the optimal theta coefficients
    theta_opt = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
    return theta_opt
    pass

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=0.01, size=n_features)
    pass

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
        # 1. Compute current predictions across all samples using our Step 007 logic
    y_pred = predict_linear(X, weights)
    
    # 2. Calculate the analytical MSE gradient vector using our Step 009 logic
    gradient = mse_gradient(X, y, y_pred)
    
    # 3. Take an analytical step down the error surface
    updated_weights = weights - lr * gradient
    
    return updated_weights
    pass

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
        # Compute predictions using your verified Step 007 engine
    preds_train = predict_linear(X_train, weights)
    preds_val = predict_linear(X_val, weights)
    
    # Compute loss using your verified Step 008 engine
    train_loss = mse_loss(y_train, preds_train)
    y_val_loss = mse_loss(y_val, preds_val)
    
    return train_loss, y_val_loss
    pass

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
        # A strict improvement check (no tolerance padding required by default unless specified)
    if val_loss < best_val_loss:
        updated_best_val_loss = val_loss
        updated_wait = 0
        updated_best_weights = weights.copy() # Secure a fresh parameter checkpoint
    else:
        updated_best_val_loss = best_val_loss
        updated_wait = wait + 1
        updated_best_weights = best_weights   # Retain our previous best checkpoint array
        
    # Check if our incremental wait counter has exhausted the patience limit [source: 6, 1.3.1]
    stop_training = updated_wait >= patience
    
    return updated_best_val_loss,updated_wait, updated_best_weights,stop_training
    pass

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
        # 1. Generate the initial random weights vector using our verified Step 011 engine
    initial_weights = initialize_weights(n_features, seed=seed)
    
    # 2. Return the exact dictionary layout and string keys required by the platform
    return {
        "weights": initial_weights.copy().astype(float),
        "best_weights": initial_weights.copy().astype(float),
        "best_val_loss": float(np.inf),
        "wait": int(0),
        "train_losses": [],
        "val_losses": [],
        "stopped": False
    }
    pass

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """
    Advances training by a single full-batch epoch using the mutable state dict.
    Unpacks Step 014 in the exact sequence it returns.
    """
    # 1. Take one GD step using our current state weights
    updated_weights = gd_step(X_train, y_train, state["weights"], lr)
    
    # 2. Extract cross-validation losses for this pass
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, updated_weights)
    
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)
    
    # 3. Unpack step 014 using its exact return signature sequence:
    # (stop_training, updated_best_val_loss, updated_wait, updated_best_weights)
    b_loss, wait_cnt, b_w, stop = update_early_stop_state(
        val_loss, state["best_val_loss"], state["wait"], 
        updated_weights, state["best_weights"], patience
    )
    
    # 4. Commit all synchronized changes directly back into the mutable state dictionary
    state["weights"] = updated_weights.copy()
    state["best_weights"] = b_w.copy() # Safe to call .copy() now because b_w is an array
    state["best_val_loss"] = b_loss
    state["wait"] = wait_cnt
    state["stopped"] = stop
    
    return state

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

