import os
import json

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import lib.datasets
import lib.utils
import lib.dataset
import lib.model
import lib.training



def run_experiments(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    lib.utils.set_seed(config.seed)


    train_datasets = list()
    for ds in config.train_datasets:
        __import__(f'lib.datasets.{ds}')
        df = lib.datasets.__dict__[ds].load_df()
        df['dataset'] = ds
        df['patient_id'] = df['dataset'] + '_' + df['patient_id'].apply(str)
        train_datasets.append(df)

    df = pd.concat(train_datasets, ignore_index=True)

    patients = df['patient_id'].unique()

    train_patients, test_patients = train_test_split(patients, test_size=config.test_size, random_state=config.seed)
    train_patients, valid_patients = train_test_split(train_patients, test_size=config.valid_size, random_state=config.seed)

    train = df[df['patient_id'].isin(train_patients)]
    valid = df[df['patient_id'].isin(valid_patients)]
    test = df[df['patient_id'].isin(test_patients)]


    train_classes =  lib.utils.calsses_from_captions(train['label'].values, threshold=config.min_class_count)
    valid_classes =  lib.utils.calsses_from_captions(valid['label'].values, threshold=config.min_class_count)
    test_classes = lib.utils.calsses_from_captions(test['label'].values, threshold=config.min_class_count)
   
    train_classes = [class_ for class_ in train_classes if class_ not in config.excluded_classes]
    train_classes = [class_ for class_ in train_classes if class_ != config.normal_class]
    valid_classes = [class_ for class_ in valid_classes if class_ in train_classes]
    test_classes = [class_ for class_ in test_classes if class_ in train_classes]
    
    test_classes, zero_shot_classes = train_test_split(test_classes, test_size=config.zero_shot_classes_size, random_state=config.seed)  
    train_classes = [class_ for class_ in train_classes if class_ not in zero_shot_classes]
    valid_classes = [class_ for class_ in valid_classes if class_ not in zero_shot_classes]
 
    train_classes.append(config.normal_class)
    valid_classes.append(config.normal_class)
    test_classes.append(config.normal_class)

    
    train_classes = sorted(train_classes)
    valid_classes = sorted(valid_classes)
    test_classes = sorted(valid_classes)
    zero_shot_classes = sorted(zero_shot_classes)
    
    print('Train/valid/test classes counts:', len(train_classes), len(valid_classes), len(test_classes), len(zero_shot_classes))

    train = train.copy()
    valid = valid.copy()
    
    train['label'] = lib.utils.remove_classes(zero_shot_classes, train['label'].to_list())
    valid['label'] = lib.utils.remove_classes(zero_shot_classes, valid['label'].to_list())


    config.train_classes = train_classes
    config.valid_classes = valid_classes
    config.test_classes = test_classes
    config.zero_shot_classes = zero_shot_classes
    
    train_ds = lib.dataset.CLIP_ECG_Dataset(train, config)
    valid_ds = lib.dataset.CLIP_ECG_Dataset(valid, config)
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    
    net = lib.model.CLIPModel(config)
    net = net.to(config.device)
    params = [
        {"params": net.image_encoder.parameters(), "lr": config.image_encoder_lr},
        {"params": net.image_projection.parameters(), "lr": config.head_lr},
        {"params": net.text_projection.parameters(), "lr": config.head_lr},
    ]
    
    optimizer = torch.optim.Adam(params)
    
    history = list()
    best_valid_score = 0.0
    for epoch in range(config.epochs):
        print(f"Epoch: {epoch + 1}")
        hrow = dict()
        hrow['epoch'] = epoch
        net.train()
        train_loss_meter, train_accuracy_meter = lib.training.train_epoch(net, train_dl, optimizer, train_classes, config)
        hrow['train_loss'] = train_loss_meter.avg
        
        metrics = lib.training.valid_epoch(net, train_dl, train_classes, config) 
        hrow.update({f'train_{key}': val for key, val in metrics.items()})
        #hrow['train_mean_rocaucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_rocauc') and val is not None])
        #hrow['train_mean_praucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_prauc') and val is not None])
        print('Train:', hrow['train_mean_rocaucs'], hrow['train_mean_praucs'])
        
        metrics = lib.training.valid_epoch(net, valid_dl, valid_classes, config) 
        hrow.update({f'valid_{key}': val for key, val in metrics.items()})
        #hrow['valid_mean_rocaucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_rocauc') and val is not None])
        #hrow['valid_mean_praucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_prauc') and val is not None])
        print('Valid:', hrow['valid_mean_rocaucs'], hrow['valid_mean_praucs'])
        
        history.append(hrow)
        pd.DataFrame(history).to_csv(config.logs_path + f'/{config.name}.csv', index=False)
    
        if hrow['valid_mean_rocaucs'] > best_valid_score:
            best_valid_score = hrow['valid_mean_rocaucs']
            torch.save(net.state_dict(), config.models_path + f'/{config.name}.pt')      

    net = lib.model.CLIPModel(config)
    net.load_state_dict(torch.load(config.models_path + f'/{config.name}.pt', weights_only=True))
    net.to(config.device)

    config.test_metrics = dict()
    config.zero_shot_test_metrics = dict()
    
    for test_ds_name in test['dataset'].unique():
        test_subset = test[test['dataset'] == test_ds_name]
    
        test_ds = lib.dataset.CLIP_ECG_Dataset(test_subset, config)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
        metrics = lib.training.valid_epoch(net, test_dl, config.test_classes, config) 
        config.test_metrics[test_ds_name] = metrics
    
        metrics = lib.training.valid_epoch(net, test_dl, config.zero_shot_classes, config) 
        config.zero_shot_test_metrics[test_ds_name] = metrics

    config.exp2_classes = dict()
    config.exp2_trained_classes = dict()
    config.exp2_untrained_classes = dict()
    config.exp2_metrics_trained = dict()
    config.exp2_metrics_untrained = dict()
    
    for exp2_ds_name in config.test_datasets:
        
        __import__(f'lib.datasets.{exp2_ds_name}')
        df = lib.datasets.__dict__[exp2_ds_name].load_df()
        df['dataset'] = exp2_ds_name
        df['patient_id'] = df['dataset'] + '_' + df['patient_id'].apply(str)
      
        config.exp2_classes[exp2_ds_name] = lib.utils.calsses_from_captions(df['label'].values, threshold=config.min_class_count)
        config.exp2_trained_classes[exp2_ds_name] = list(set(config.exp2_classes[exp2_ds_name]) & set(config.train_classes))
        config.exp2_untrained_classes[exp2_ds_name] = list(set(config.exp2_classes[exp2_ds_name]) - set(config.train_classes))

        test_ds = lib.dataset.CLIP_ECG_Dataset(df, config)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
        metrics = lib.training.valid_epoch(net, test_dl, config.exp2_trained_classes[exp2_ds_name], config) 
        config.exp2_metrics_trained[exp2_ds_name] = metrics

        metrics = lib.training.valid_epoch(net, test_dl, config.exp2_untrained_classes[exp2_ds_name], config) 
        config.exp2_metrics_untrained[exp2_ds_name] = metrics
 
    cfg = {k:v for k, v in config.__dict__.items() if not k.startswith('__')}
    with open(f'{config.logs_path}/{config.name}.cfg', 'w') as fp:
        json.dump(cfg, fp)