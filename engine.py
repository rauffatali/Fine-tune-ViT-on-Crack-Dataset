import os
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score

from utils.eval_utils import classification_report
from plots import plot_metrics_curve, plot_conf_matrix, plot_precision_recall_curve, plot_roc_curve

from typing import Optional, Callable, Tuple, Dict

BAR_FORMAT = '{desc}: {percentage:3.0f}%| {bar} | {n_fmt}/{total_fmt} {unit}'

def train_one_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    verbose: bool) -> Tuple[float, float, float, float, float]:

    # Set the model to train mode
    model.train()

    y_pred = []
    y_true = []

    running_loss = 0.0
    with tqdm(total=len(train_loader), desc="\tTrain", unit='', bar_format=BAR_FORMAT, disable=not verbose) as pbar:
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(torch.exp(outputs), 1)
            y_pred.extend(preds.data.cpu().numpy()) # Save Predictions
            y_true.extend(labels.data.cpu().numpy()) # Save Actual

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item() * inputs.size(0)

            pbar.update()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(y_true, y_pred)
        train_precision = precision_score(y_true, y_pred)
        train_recall = recall_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred)

        pbar.unit = f'| Loss: {train_loss:.4f} | Accuracy: {train_acc*100:.2f}% ' \
                    f'| Precision: {train_precision*100:.2f}% | Recall: {train_recall*100:.2f}% ' \
                    f'| F1 score: {train_f1*100:.2f}%'

    return train_loss, train_acc, train_precision, train_recall, train_f1

def eval(
    model: nn.Module, 
    val_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    device: torch.device, 
    verbose: bool,  
    evaluate: bool = False,
    print_report: bool = False) -> Tuple[float, float, float, float, float]:

    # Set the model to evaluation mode
    model.eval()  

    y_pred = []
    y_score = []
    y_true = []
    running_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="\tValidation", unit='', bar_format=BAR_FORMAT, disable=not verbose) as pbar:
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(torch.exp(outputs), 1)
                y_pred.extend(preds.data.cpu().numpy())
                y_score.extend(outputs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

                # Calculate loss
                loss = criterion(outputs, labels)

                # Update the running loss
                running_loss += loss.item() * inputs.size(0)
            
                pbar.update()
                
            val_loss = running_loss / len(val_loader.dataset)
            val_acc = accuracy_score(y_true, y_pred)
            val_precision = precision_score(y_true, y_pred)
            val_recall = recall_score(y_true, y_pred)
            val_f1 = f1_score(y_true, y_pred)

            pbar.unit = f'| Loss: {val_loss:.4f} | Accuracy: {val_acc*100:.2f}% ' \
                        f'| Precision: {val_precision*100:.2f}% | Recall: {val_recall*100:.2f}% ' \
                        f'| F1 score: {val_f1*100:.2f}%'

    if evaluate:
        if print_report:        
            print(classification_report(y_true, y_pred))

        return y_true, y_pred, y_score

    else:
        return val_loss, val_acc, val_precision, val_recall, val_f1

def train(
    model: nn.Module, 
    NUM_EPOCHS: int, 
    dataloaders: Dict[str, DataLoader], 
    device: torch.device, 
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    optimizer: torch.optim.Optimizer, 
    scheduler=None, 
    early_stopping=None, 
    checkpoint_interval: Optional[int] = None, 
    multiple_gpu: bool = False,
    save: bool = True,
    save_dir: Path = None,
    verbose: bool = True) -> dict:

    train_hist = {
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': []
    }

    if save:
        # Init training params
        hyps = {
            'criterion': type(criterion).__name__,
            'optimizer': type(optimizer).__name__, 
        }

        if hyps['optimizer'] == 'SGD':
            hyps['opt_learning_rate'] = optimizer.param_groups[0]['lr'] 
            hyps['opt_momentum'] = optimizer.param_groups[0]['momentum']
            hyps['opt_weight_decay'] = optimizer.param_groups[0]['weight_decay']
        
        if hyps['optimizer'] == 'Adam':
            hyps['opt_learning_rate'] = optimizer.param_groups[0]['lr'] 
            hyps['opt_weight_decay'] = optimizer.param_groups[0]['weight_decay']
        
        if scheduler:
            hyps['scheduler'] = type(scheduler).__name__
            hyps['scheduler_factor'] = scheduler.factor
            hyps['scheduler_patience'] = scheduler.patience
            hyps['scheduler_threshold'] = scheduler.threshold

        opts = {
            'epochs': NUM_EPOCHS,
            'dataset': dataloaders['train'].dataset.dataset_name,
            'dataset_size': len(dataloaders['train'].dataset) + len(dataloaders['valid'].dataset) + len(dataloaders['test'].dataset),
            'batch_size': dataloaders['train'].batch_size,
            'imgsz': dataloaders['train'].dataset.transform.transforms[0].size,
            'model_name': model.model_name,
            'pretrained': model.pretrained,
            'trainable_layers': model.trainable_layers,
            'device': device,
            'multiple_gpu': multiple_gpu,
            'hyps': hyps,          
        }
        
        # Init experiments folder
        if save_dir is None:
            now = datetime.now().strftime("%Y-%m-%d %H-%M")
            save_dir = f"experiments/train/({now}) {opts['model_name']}_on_{opts['dataset']}_w_epoch_{opts['epochs']}_lr_{hyps['opt_learning_rate']}"
        os.makedirs(save_dir, exist_ok=True)

        # Save training params
        opts_path = os.path.join(save_dir, 'opt.yaml')
        with open(opts_path, 'w') as f:
            yaml.dump(opts, f, sort_keys=False)
    
    # Train in multiple gpus
    if multiple_gpu:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):

        if verbose:
            lr = optimizer.param_groups[0]['lr']
            memory_usage = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | Learning Rate: {lr} | mem: {memory_usage}')

        train_loss, train_acc, train_precision, train_recall, _ = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device, verbose)
        val_loss, val_acc, val_precision, val_recall, _ = eval(model, dataloaders['valid'], criterion, device, verbose)
            
        train_hist['train_loss'].append(train_loss)
        train_hist['train_accuracy'].append(train_acc)
        train_hist['train_precision'].append(train_precision)
        train_hist['train_recall'].append(train_recall)
        
        train_hist['val_loss'].append(val_loss)
        train_hist['val_accuracy'].append(val_acc)
        train_hist['val_precision'].append(val_precision)
        train_hist['val_recall'].append(val_recall)

        # Adjust learning rate
        if scheduler:
            if hyps['scheduler']=='ReduceLROnPlateau':
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
        
        if save:
            # Save best model
            weights_dir = os.path.join(save_dir, 'weights') 
            os.makedirs(weights_dir, exist_ok=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(weights_dir, 'best.pt'))
        
            # Save last model
            torch.save(model.state_dict(), os.path.join(weights_dir, 'last.pt'))

            # Save training epoch    
            csv_path = os.path.join(save_dir, 'history.csv')

            fields = ['epoch'] + list(train_hist.keys())
            data = [epoch+1, 
                    train_loss, train_acc, train_precision, train_recall, 
                    val_loss, val_acc, val_precision, val_recall]

            header = ','.join(['%20s' % field for field in fields]) + '\n'
            s = '' if os.path.exists(csv_path) else header # add header
            with open(csv_path, 'a') as f:
                row = ','.join(['%20.5g' % d for d in data])
                f.write(s + row + '\n')
        
        # Check early stop
        if early_stopping is not None:
            if early_stopping(val_loss):
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

    if save:
        plot_metrics_curve(train_hist, plot=False, save=True, save_dir=save_dir)

        y_true, y_pred, y_score = eval(model, dataloaders['valid'], criterion, device, verbose=False, evaluate=True)
        plot_conf_matrix(y_true, y_pred, classes=['non-cracked', 'cracked'], plot=False, save=True, save_dir=save_dir)
        plot_precision_recall_curve(y_true, y_score, plot=False, save=True, save_dir=save_dir)
        plot_roc_curve(y_true, y_score, plot=False, save=True, save_dir=save_dir)
    
    return train_hist