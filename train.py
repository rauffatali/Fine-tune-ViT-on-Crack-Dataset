import os
import yaml
from datetime import datetime
from collections import OrderedDict

import torch
from sklearn.metrics import accuracy_score

def train_one_epoch(model, dataloaders, criterion, optimizer, device):
    # Set the model to train mode
    model.train()

    # Initialize the predicted and actual values
    y_pred = []
    y_true = []

    # Initialize the running loss
    running_loss = 0.0

    # Iterate over the batches of the train loader
    for inputs, labels in dataloaders['train']:
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, preds = (torch.max(torch.exp(outputs), 1))
        y_pred.extend(preds.data.cpu().numpy()) # Save Predictions
        y_true.extend(labels.data.cpu().numpy()) # Save Actual

        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Zero the optimizer gradients
        optimizer.zero_grad()

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item() * inputs.size(0)

    # Calculate the train loss, accuracy, precision, recall, and f1 score
    train_loss = running_loss / len(dataloaders['train'].dataset)
    train_acc = accuracy_score(y_true, y_pred)

    return train_loss, train_acc

def eval(model, dataloaders, criterion, device):
    # Set the model to evaluation mode
    model.eval()

    # Initialize the predicted and actual values
    y_pred = []
    y_true = []
    
    # Initialize the running loss and accuracy
    running_loss = 0.0

    # Iterate over the batches of the validation loader
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = (torch.max(torch.exp(outputs), 1))
            y_pred.extend(preds.data.cpu().numpy()) # Save Predictions
            y_true.extend(labels.data.cpu().numpy()) # Save Actual

            # Calculate loss
            loss = criterion(outputs, labels)

            # Update the running loss
            running_loss += loss.item() * inputs.size(0)

    # Calculate the validation loss, accuracy, precision, recall, and f1 score
    val_loss = running_loss / len(dataloaders['valid'].dataset)
    val_acc = accuracy_score(y_true, y_pred)

    return val_loss, val_acc, y_pred, y_true

def train(model, NUM_EPOCHS, dataloaders, criterion, optimizer, scheduler, device, checkpoint_interval=None, verbose=True):

    # Initialize training params
    params = {
        'num_epochs': NUM_EPOCHS,
        'batch_size': dataloaders['train'].batch_size,
        'dataset': dataloaders['train'].dataset.dataset_name,
        'dataset_size': len(dataloaders['train'].dataset) + len(dataloaders['valid'].dataset) + len(dataloaders['test'].dataset),
        'model_name': model.name,
        'criterion': type(criterion).__name__,
        'optimizer': type(optimizer).__name__,
        'learning_rate': optimizer.param_groups[0]['lr'], 
        'momentum': optimizer.param_groups[0]['momentum'],
    }

    # Initialize experiment folder
    # now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = f"train/experiment_{params['model_name']}_on_{params['dataset']}_w_epoch_{params['num_epochs']}_lr_{params['learning_rate']}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training params
    params_path = os.path.join(save_dir, 'params.yaml')
    with open(params_path, 'w') as f:
        yaml.dump(params, f)

    train_hist = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    best_val_loss = float('inf')

    # Train the model for the specified number of epochs
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, dataloaders, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval(model, dataloaders, criterion, device)

        if verbose:
            # Print the epoch results
            print('Epoch [{}/{}], lr: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
                  .format(epoch+1, NUM_EPOCHS, optimizer.param_groups[0]['lr'], train_loss, train_acc, val_loss, val_acc))
            
        train_hist['train_losses'].append(train_loss)
        train_hist['train_accs'].append(train_acc)
        
        train_hist['val_losses'].append(val_loss)
        train_hist['val_accs'].append(val_acc)

        # Adjust learning rate
        scheduler.step(val_loss)
        
        # Save best model
        weights_dir = os.path.join(save_dir, 'weights') 
        os.makedirs(weights_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, 'best.pt'))

        # Save last model
        torch.save(model.state_dict(), os.path.join(weights_dir, 'last.pt'))

        if checkpoint_interval:
            # Save checkpoint
            checkpoints_dir = os.path.join(save_dir, 'checkpoints') 
            os.makedirs(checkpoints_dir, exist_ok=True)
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_state = OrderedDict([
                    ('model_state_dict', model.state_dict()),
                    ('optimizer_state_dict', optimizer.state_dict()),
                    ('epoch', epoch + 1),
                ])
                torch.save(checkpoint_state, os.path.join(checkpoints_dir, f'checkpoint_{epoch+1}.pt'))

    return train_hist