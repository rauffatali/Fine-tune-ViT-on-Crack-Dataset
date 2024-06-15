class EarlyStopper:
    """
    Custom Early Stopping implementation.

    Args:
        patience (int): Number of epochs with no improvement to wait before stopping.
        mode (str, optional): Mode for monitoring the validation metric ('min' or 'max'). Defaults to 'min'.
    """
    def __init__(self, patience: int, min_delta: int = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False