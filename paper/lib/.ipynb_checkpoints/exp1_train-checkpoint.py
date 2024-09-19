from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
import json

import os

from lib import utils, model, dataset, training, codes


def exp1_train(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    utils.set_seed(config.seed)
    df = utils.get_data_cached(config.data_path, codes.DECODE_DICT, config.cache_path + '/df.csv')
    
    train, test = train_test_split(df, test_size=config.test_size, random_state=config.seed)
    train, valid = train_test_split(train, test_size=config.valid_size, random_state=config.seed)
     
    train_classes =  utils.calsses_from_captions(train['label'].values, threshold=config.min_class_count)
    valid_classes =  utils.calsses_from_captions(valid['label'].values, threshold=config.min_class_count)
    test_classes = utils.calsses_from_captions(test['label'].values, threshold=config.min_class_count)
    
    valid_classes = [class_ for class_ in valid_classes if class_ in train_classes]
    test_classes = [class_ for class_ in test_classes if class_ in train_classes]
    
    print('Train/valid/test classes counts:', len(train_classes), len(valid_classes), len(test_classes))
    
    config.train_classes = train_classes
    config.valid_classes = valid_classes
    config.test_classes = test_classes
    
    train_ds = dataset.CLIP_ECG_Dataset(train, config)
    valid_ds = dataset.CLIP_ECG_Dataset(valid, config)
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    
    net = model.CLIPModel(config)
    net = net.to(config.device)
    params = [
        {"params": net.image_encoder.parameters(), "lr": config.image_encoder_lr},
        {"params": net.image_projection.parameters(), "lr": config.head_lr},
        {"params": net.text_projection.parameters(), "lr": config.head_lr},
    ]
    
    optimizer = torch.optim.Adam(params)
    
    cfg = {k:v for k, v in config.__dict__.items() if not k.startswith('__')}
    cfg_hash = utils.generate_dict_hash(cfg)
    
    with open(f'{config.logs_path}/{cfg_hash}.cfg', 'w') as fp:
        json.dump(cfg, fp)
    
    history = list()
    best_valid_score = 0.0
    for epoch in range(config.epochs):
        print(f"Epoch: {epoch + 1}")
        hrow = dict()
        hrow['epoch'] = epoch
        net.train()
        train_loss_meter, train_accuracy_meter = training.train_epoch(net, train_dl, optimizer, train_classes, config)
        hrow['train_loss'] = train_loss_meter.avg
        
        metrics = training.valid_epoch(net, train_dl, train_classes, config) 
        hrow.update({f'train_{key}': val for key, val in metrics.items()})
        #hrow['train_mean_rocaucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_rocauc') and val is not None])
        #hrow['train_mean_praucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_prauc') and val is not None])
        print('Train:', hrow['train_mean_rocaucs'], hrow['train_mean_praucs'])
        
        metrics = training.valid_epoch(net, valid_dl, valid_classes, config) 
        hrow.update({f'valid_{key}': val for key, val in metrics.items()})
        #hrow['valid_mean_rocaucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_rocauc') and val is not None])
        #hrow['valid_mean_praucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_prauc') and val is not None])
        print('Valid:', hrow['valid_mean_rocaucs'], hrow['valid_mean_praucs'])
        
        history.append(hrow)
        pd.DataFrame(history).to_csv(config.logs_path + f'/{cfg_hash}.csv', index=False)
    
        if hrow['valid_mean_rocaucs'] > best_valid_score:
            best_valid_score = hrow['valid_mean_rocaucs']
            torch.save(net.state_dict(), config.models_path + f'/{cfg_hash}.pt')      