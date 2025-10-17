def linear_anneal(epoch: int, start_epoch: int, end_epoch: int, start_val: float, end_val: float) -> float:
    """Performs linear annealing."""
    if start_epoch >= end_epoch: return end_val
    if epoch < start_epoch: return start_val
    if epoch >= end_epoch: return end_val
    return start_val + (end_val - start_val) * (epoch - start_epoch) / (end_epoch - start_epoch)