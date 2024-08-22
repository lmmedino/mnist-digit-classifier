import torch.optim as optim

def get_adam_optimizer(model, lr=0.001, weight_decay=0.0001):
    """
    This function returns an Adam optimizer with L2 regularization for the given model.

    Args:
        model: The model whose parameters will be optimized.
        lr: The learning rate for the optimizer.
        weight_decay: The L2 regularization coefficient (default is 0.0001).

    Returns:
        optimizer: An Adam optimizer with L2 regularization.
    """
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
