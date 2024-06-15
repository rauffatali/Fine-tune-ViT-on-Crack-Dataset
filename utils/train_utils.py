class EarlyStopping:
    """
    Early stopping to monitor a validation metric (e.g., loss) during training and stop 
    training if the metric fails to improve for a specified number of epochs.

    Args:
        patience (int, optional): Number of epochs with no improvement to wait before stopping. 
                                  Defaults to 1.
        min_delta (float, optional): Minimum change in the monitored metric considered an improvement. 
                                    Defaults to 0, meaning any decrease in loss is considered an improvement.
        mode (str, optional): Mode for monitoring the validation metric 
                              ('min' for minimizing loss or 'max' for maximizing accuracy). 
                              Defaults to 'min'.
    """
    def __init__(self, patience=1, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.counter = 0

        if self.mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")
    
        self._best_val = float('inf') if self.mode == 'min' else float('-inf')

    def early_stop(self, validation_metric):
        """
        Checks if early stopping should be triggered based on the current validation metric.

        Args:
            validation_metric (float): Current validation metric value (e.g., loss or accuracy).

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.mode == 'min' and validation_metric < self._best_val:
            self._best_val = validation_metric
            self.counter = 0
        elif self.mode == 'max' and validation_metric > self._best_val:
            self._best_val = validation_metric
            self.counter = 0
        elif validation_metric + self.min_delta <= self._best_val:
            self.counter += 1
        if self.counter >= self.patience:
            return True
        return False

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs