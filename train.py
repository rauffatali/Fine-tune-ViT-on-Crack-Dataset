import torch

def train_one_epoch(model, dataloaders, criterion, optimizer, device):
    # Set the model to train mode
    model.train()

    # Initialize the running loss and accuracy
    batch_loss = 0.0
    batch_corrects = 0

    # Iterate over the batches of the train loader
    for inputs, labels in dataloaders['train']:
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Zero the optimizer gradients
        optimizer.zero_grad()

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Update the running loss and accuracy
        batch_loss += loss.item() * inputs.size(0)
        batch_corrects += torch.sum(preds == labels.data)

    # Calculate the train loss and accuracy
    train_dataset_len = dataloaders['train'].dataset.__len__()
    train_loss = batch_loss / train_dataset_len
    train_acc = batch_corrects.double() / train_dataset_len

    # Set the model to evaluation mode
    model.eval()

    # Initialize the running loss and accuracy
    batch_loss = 0.0
    batch_corrects = 0

    # Iterate over the batches of the validation loader
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Update the running loss and accuracy
            batch_loss += loss.item() * inputs.size(0)
            batch_corrects += torch.sum(preds == labels.data)

    # Calculate the validation loss and accuracy
    valid_dataset_len = dataloaders['valid'].dataset.__len__()
    val_loss = batch_loss / valid_dataset_len
    val_acc = batch_corrects.double() / valid_dataset_len

    return train_loss, val_loss, train_acc, val_acc