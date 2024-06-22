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
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.counter = 0

        if self.mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")
    
        self._best_val = float('inf') if self.mode == 'min' else float('-inf')

    def __call__(self, validation_metric):
        """
        Checks if early stopping should be triggered based on the current validation metric.

        Args:
            validation_metric (float): Current validation metric value (e.g., loss or accuracy).

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.mode == 'min':
            if validation_metric < self._best_val:
                self._best_val = validation_metric
                self.counter = 0
            elif (validation_metric + self.min_delta) >= self._best_val:
                self.counter += 1
        elif self.mode == 'max':
            if validation_metric > self._best_val:
                self._best_val = validation_metric
                self.counter = 0
            elif (validation_metric + self.min_delta) <= self._best_val:
                self.counter += 1

        if self.counter >= self.patience:
            return True
        
        return False
    
if __name__ == '__main__':

    early_stopping = EarlyStopping(patience=5, min_delta=0.01, mode='max')

    # Hypothetical validation metric values (e.g., acc)
    accs = [0.52, 0.48, 0.49, 0.45, 0.51, 0.48, 0.49]
    # Hypothetical validation metric values (e.g., loss)
    # losses = [0.52, 0.51, 0.5, 0.49, 0.5, 0.51, 0.5, 0.52, 0.53]

    for i, m in enumerate(accs):
        stop = early_stopping(m)
        print(f'Epoch {i+1}, Loss {m}, Counter: {early_stopping.counter}, Patience: {early_stopping.patience}')
        if stop:
            print(f"Early stopping triggered at epoch {i+1}!")
            break