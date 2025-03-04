from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
import sys


def calc_metrics(image_embeddings, captions, class_embeddings, class_names):
    similarity = nxn_cos_sim(image_embeddings, class_embeddings, dim=1)
    predictions_ids = similarity.argmax(dim=1)
    predictions = [class_names[idx] for idx in predictions_ids]
    tps = [prediction in caption for prediction, caption in zip(predictions, captions)]
    accuracy = np.mean(tps)
    
    results = dict()
    results['accuracy'] = accuracy
    
    similarity = similarity.detach().cpu().numpy()
    for i, name in enumerate(class_names):
        
        true = np.array([name in caption for caption in captions]).astype('int32')
        results[f'{name}_trues'] = int(true.sum())
        results[f'{name}_samples'] = int(true.shape[0])
        results[f'{name}_freq'] = float(true.sum() / true.shape[0])
        if true.std() > 0:
            results[f'{name}_rocauc'] = roc_auc_score(true, similarity[:, i])
            results[f'{name}_prauc'] = average_precision_score(true, similarity[:, i])
        else:
            results[f'{name}_rocauc'] = None
            results[f'{name}_prauc'] = None          

    vals = [val for key, val in results.items() if key.endswith('_rocauc') and val is not None]
    if len(vals) > 0:
        results['mean_rocaucs'] = np.mean(vals)
    else:
        results['mean_rocaucs'] = None

    vals = [val for key, val in results.items() if key.endswith('_prauc') and val is not None]
    if len(vals) > 0:
        results['mean_praucs'] = np.mean(vals)
    else:
        results['mean_praucs'] = None
    return results 

def calc_metrics_pretrain(trues, preds, class_names):
    results = dict()
    for i, name in enumerate(class_names):
        
        true = trues[:, i]
        pred = preds[:, i]
        results[f'{name}_trues'] = int(true.sum())
        results[f'{name}_samples'] = int(true.shape[0])
        results[f'{name}_freq'] = float(true.sum() / true.shape[0])
        if true.std() > 0:
            results[f'{name}_rocauc'] = roc_auc_score(true, pred)
            results[f'{name}_prauc'] = average_precision_score(true, pred)
        else:
            results[f'{name}_rocauc'] = None
            results[f'{name}_prauc'] = None          

    results['mean_rocaucs'] = np.mean([val for key, val in results.items() if key.endswith('_rocauc') and val is not None])
    results['mean_praucs'] = np.mean([val for key, val in results.items() if key.endswith('_prauc') and val is not None])
    return results

def calc_accuracy(image_embeddings, captions, class_embeddings, class_names):
    similarity = nxn_cos_sim(image_embeddings, class_embeddings, dim=1)
    predictions_ids = similarity.argmax(dim=1)
    predictions = [class_names[idx] for idx in predictions_ids]
    tps = [prediction in caption for prediction, caption in zip(predictions, captions)]
    accuracy = np.mean(tps)
    return accuracy

def nxn_cos_sim(A, B, dim=1):
    a_norm = F.normalize(A, p=2, dim=dim)
    b_norm = F.normalize(B, p=2, dim=dim)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def train_epoch(model, loader, optimizer, classes, config):
    tqdm_object = tqdm(loader, total=len(loader), desc='Train', file=sys.stdout)
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    for batch in tqdm_object:
        model.train()
        batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        loss, image_embeddings, text_embeddings = model(batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        model.eval()
        with torch.no_grad():
            class_embeddings = model.text_to_embeddings(classes)
            
        accuracy = calc_accuracy(image_embeddings, batch['caption'], class_embeddings, classes)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        accuracy_meter.update(accuracy, count)
        
    return loss_meter, accuracy_meter


def valid_epoch(model, loader, classes, config):
    model.eval()
   
    with torch.no_grad():
        class_embeddings = model.text_to_embeddings(classes).detach().cpu()
    
    tqdm_object = tqdm(loader, total=len(loader), desc='Eval', file=sys.stdout)
    embeddings = list()
    captions = list()
    with torch.no_grad():
        for batch in tqdm_object:
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, image_embeddings, text_embeddings = model(batch)
            embeddings.append(image_embeddings.cpu())
            captions += batch['caption']
        
    embeddings = torch.cat(embeddings)
    metric = calc_metrics(embeddings, captions, class_embeddings, classes)
    return metric


def train_epoch_pretrain(model, loader, optimizer, classes, config):
    losses = 0
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    trues = list()
    preds = list()
    for batch in tqdm(loader, desc='Train', file=sys.stdout): 

        ecgs = batch['image'].to(config.device)
        true = batch['targets'].to(config.device)
        pred = model(ecgs)
        loss = criterion(pred, true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        trues.append(true)
        preds.append(pred)

    losses /= len(loader)

    trues = torch.cat(trues).detach().cpu().numpy()
    preds = torch.cat(preds).detach().cpu().numpy()

    metrics = calc_metrics_pretrain(trues, preds, classes)

    return losses, metrics

def valid_epoch_pretrain(model, loader, classes, config):
    losses = 0
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    trues = list()
    preds = list()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', file=sys.stdout):
            ecgs = batch['image'].to(config.device)
            true = batch['targets'].to(config.device)
            pred = model(ecgs)
            loss = criterion(pred, true)
            losses += loss.item()

            trues.append(true)
            preds.append(pred)
            
    losses /= len(loader)
    
    trues = torch.cat(trues).detach().cpu().numpy()
    preds = torch.cat(preds).detach().cpu().numpy()
    metrics = calc_metrics_pretrain(trues, preds, classes)
    return losses, metrics