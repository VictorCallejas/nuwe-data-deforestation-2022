import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn import metrics

import numpy as np 
import pandas as pd

def train_epoch(model, train_dataloader, device, criterion, optimizer, scheduler=None):
    '''Performs a train epoch
    '''

    total_train_loss = 0
    model.train()

    logits = []
    ground_truth = []

    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        
        b_image = batch[0].to(device)
        b_year = batch[1].to(device)
        b_neightbours_context = batch[2].to(device)
        b_labels = batch[3]
        
        with torch.cuda.amp.autocast(enabled=False):
            b_logits = model(b_image, b_year, b_neightbours_context).cpu()
                
        loss = criterion(b_logits.squeeze(),b_labels)
        loss.backward()

        total_train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        logits.extend(b_logits.detach().numpy())
        ground_truth.extend(b_labels.numpy())

    logits = np.array(logits)
    y_probas = nn.Softmax(dim=1)(torch.Tensor(logits))
    y_labels = np.argmax(y_probas, axis = 1)

    train_f1 = metrics.f1_score(ground_truth, y_labels, average='macro')
    train_acc = metrics.accuracy_score(ground_truth, y_labels)
    train_f1_no_average = metrics.f1_score(ground_truth, y_labels, average=None)
    avg_train_loss = total_train_loss/len(train_dataloader)
    
    scheduler.step()

    return train_f1, train_acc, train_f1_no_average, avg_train_loss

def valid_epoch(model,valid_dataloader,device,criterion):
    '''Performs a validation epoch
    '''
            
    model.eval()

    total_dev_loss = 0

    logits = []
    ground_truth = []
    
    for step, batch in enumerate(valid_dataloader):
        
        b_image = batch[0].to(device)
        b_year = batch[1].to(device)
        b_neightbours_context = batch[2].to(device)
        b_labels = batch[3]
        
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad(): 
                b_logits = model(b_image, b_year, b_neightbours_context).cpu()

        loss = criterion(b_logits.squeeze(),b_labels)
        total_dev_loss += loss.item()
        
        logits.extend(b_logits.detach().numpy())
        ground_truth.extend(b_labels.numpy())

    logits = np.array(logits)
    y_probas = nn.Softmax(dim=1)(torch.Tensor(logits))
    y_labels = np.argmax(y_probas, axis = 1)

    valid_f1 = metrics.f1_score(ground_truth, y_labels, average='macro')
    valid_f1_no_average = metrics.f1_score(ground_truth, y_labels, average=None)
    valid_acc = metrics.accuracy_score(ground_truth, y_labels)
    avg_valid_loss = total_dev_loss/len(valid_dataloader)

    return valid_f1, valid_acc, valid_f1_no_average, avg_valid_loss

def generate_test_results(model, test_dataloader, device):
    ''' Generates test results
    '''

    model.eval()
    
    image_ids, labels = [], []
    
    for step, batch in enumerate(test_dataloader):
        
        b_image = batch[0].to(device)
        b_year = batch[1].to(device)
        b_neightbours_context = batch[2].to(device)
        b_img_ids = batch[3] # Image IDs
        
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad(): 
                b_logits = model(b_image, b_year, b_neightbours_context).cpu()

        b_probas = nn.Softmax(dim=1)(b_logits).detach().numpy()
        b_labels = np.argmax(b_probas, axis = 1)
        
        labels.extend(b_labels)
        image_ids.extend(b_img_ids)

    predictions = pd.Series(labels, index=image_ids)

    return predictions