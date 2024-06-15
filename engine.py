import os
import time
import yaml
from datetime import datetime
from collections import OrderedDict

import torch
from sklearn.metrics import accuracy_score, classification_report

from utils.train_utils import epoch_time

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    # Set the model to train mode
    model.train()

    y_pred = []
    y_true = []

    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the optimizer gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        _, preds = (torch.max(torch.exp(outputs), 1))
        y_pred.extend(preds.data.cpu().numpy()) # Save Predictions
        y_true.extend(labels.data.cpu().numpy()) # Save Actual

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = accuracy_score(y_true, y_pred)

    return train_loss, train_acc

def eval(model, val_loader, criterion, device, eval_test=False):
    # Set the model to evaluation mode
    model.eval()

    y_pred = []
    y_true = []
    
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
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

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = accuracy_score(y_true, y_pred)

    if eval_test: 
        print(classification_report(y_true, y_pred))
        return y_pred, y_true
    else:
        return val_loss, val_acc

def train(model, NUM_EPOCHS, dataloaders, device, criterion, optimizer, scheduler=None, early_stopping=None, checkpoint_interval=None, multiple_gpu=False, verbose=True):

    # Init training params
    params = {
        'num_epochs': NUM_EPOCHS,
        'dataset': dataloaders['train'].dataset.dataset_name,
        'dataset_size': len(dataloaders['train'].dataset) + len(dataloaders['valid'].dataset) + len(dataloaders['test'].dataset),
        'batch_size': dataloaders['train'].batch_size,
        'model_name': model.model_name,
        'pretrained': model.pretrained,
        'trainable_layers': model.trainable_layers,
        'criterion': type(criterion).__name__,
        'optimizer': type(optimizer).__name__,        
    }

    if params['optimizer'] == 'SGD':
        params['opt_learning_rate'] = optimizer.param_groups[0]['lr'] 
        params['opt_momentum'] = optimizer.param_groups[0]['momentum']
        params['opt_weight_decay'] = optimizer.param_groups[0]['weight_decay']
    
    if params['optimizer'] == 'Adam':
        params['opt_learning_rate'] = optimizer.param_groups[0]['lr'] 
        params['opt_weight_decay'] = optimizer.param_groups[0]['weight_decay']
    
    if scheduler:
        params['scheduler'] = type(scheduler).__name__
        params['scheduler_factor'] = scheduler.factor
        params['scheduler_patience'] = scheduler.patience
        params['scheduler_threshold'] = scheduler.threshold

    # Init experiments folder
    now = datetime.now().strftime("%Y-%m-%d %H-%M")
    save_dir = f"experiments/({now}) {params['model_name']}_on_{params['dataset']}_w_epoch_{params['num_epochs']}_lr_{params['opt_learning_rate']}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training params
    params_path = os.path.join(save_dir, 'params.yaml')
    with open(params_path, 'w') as f:
        yaml.dump(params, f, sort_keys=False)

    if multiple_gpu:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    train_hist = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc = eval(model, dataloaders['valid'], criterion, device)
            
        train_hist['train_losses'].append(train_loss)
        train_hist['train_accs'].append(train_acc)
        
        train_hist['val_losses'].append(val_loss)
        train_hist['val_accs'].append(val_acc)

        # Adjust learning rate
        if scheduler:
            if params['scheduler']=='ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        if checkpoint_interval:
            checkpoints_dir = os.path.join(save_dir, 'checkpoints') 
            os.makedirs(checkpoints_dir, exist_ok=True)
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_state = OrderedDict([
                    ('model_state_dict', model.state_dict()),
                    ('optimizer_state_dict', optimizer.state_dict()),
                    ('epoch', epoch + 1),
                ])
                torch.save(checkpoint_state, os.path.join(checkpoints_dir, f'checkpoint_{epoch+1}.pt'))
        
        # Save best model
        weights_dir = os.path.join(save_dir, 'weights') 
        os.makedirs(weights_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, 'best.pt'))
        
        # Save last model
        torch.save(model.state_dict(), os.path.join(weights_dir, 'last.pt'))
        
        # Check early stop
        if early_stopping:
            if early_stopping.early_stop(val_loss):
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

        end_time = time.time()

        if verbose:
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            # print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | Epoch Time: {epoch_mins}m {epoch_secs}s')
            lr = optimizer.param_groups[0]['lr']
            memory_usage = int(torch.cuda.max_memory_allocated(device)/1024/1024)
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | Learning Rate: {lr} | mem: {memory_usage}')
            print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\tVal. Loss: {val_loss:.4f} | Val. Acc: {val_acc*100:.2f}%')

    return train_hist