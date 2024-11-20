import os
import json

import pandas as pd
import torch


import lib.utils
import lib.dataset
import lib.model
import lib.training
import lib.pretrain


def run_experiments(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    lib.utils.set_seed(config.seed)


    train_datasets = list()
    for ds in config.train_datasets:
        df = pd.read_csv(f'docs/{ds}.csv').iloc[:config.limit]
        train_datasets.append(df)

    df = pd.concat(train_datasets, ignore_index=True)

    max_fold = df['fold'].max()
    
    test_fold = config.test_fold
    valid_fold = config.valid_fold

    
    test_mask = df['fold'] == test_fold
    valid_mask = df['fold'] == valid_fold
    train_mask = (~test_mask)&(~valid_mask)
    
    train = df[train_mask].copy()
    valid = df[valid_mask].copy()
    test = df[test_mask].copy()

    print(f'Train size: {train.shape[0]}. Number of patients: {train["patient_id"].nunique()}',)
    print(f'Valid size: {valid.shape[0]}. Number of patients: {valid["patient_id"].nunique()}',)
    print(f'Test size: {test.shape[0]}. Number of patients: {test["patient_id"].nunique()}',)

    print('Preparing data:')
    train_ds = lib.dataset.CLIP_ECG_Dataset(train['ecg_file'].values, train['train_label'].values, config)
    valid_ds = lib.dataset.CLIP_ECG_Dataset(valid['ecg_file'].values, valid['train_label'].values, config)
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    
    net = lib.model.CLIPModel(config)

    if config.pretrained:
        print('Loading pretrain:')
        pretrain_name = config.name + '_pretrain'
        pretrain = lib.pretrain.ClassificationNet(config)
        pretrain.load_state_dict(torch.load(config.models_path + f'/{pretrain_name}.pt', weights_only=True))
        net.image_encoder = pretrain.encoder
    
    net = net.to(config.device)
    params = [
        {"params": net.image_encoder.parameters(), "lr": config.image_encoder_lr},
        {"params": net.image_projection.parameters(), "lr": config.head_lr},
        {"params": net.text_projection.parameters(), "lr": config.head_lr},
    ]
    
    optimizer = torch.optim.Adam(params)
    print('Main trainig loop:')
    history = list()
    best_valid_score = 0.0
    for epoch in range(config.epochs):
        print(f"Epoch: {epoch + 1}")
        hrow = dict()
        hrow['epoch'] = epoch
        net.train()
        train_loss_meter, train_accuracy_meter = lib.training.train_epoch(net, train_dl, optimizer, config.train_classes, config)
        hrow['train_loss'] = train_loss_meter.avg
        
        metrics = lib.training.valid_epoch(net, train_dl, config.train_classes, config) 
        hrow.update({f'train_{key}': val for key, val in metrics.items()})
        #hrow['train_mean_rocaucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_rocauc') and val is not None])
        #hrow['train_mean_praucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_prauc') and val is not None])
        print('Eval on train set:', hrow['train_mean_rocaucs'])
        
        metrics = lib.training.valid_epoch(net, valid_dl, config.train_classes, config) 
        hrow.update({f'valid_{key}': val for key, val in metrics.items()})
        #hrow['valid_mean_rocaucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_rocauc') and val is not None])
        #hrow['valid_mean_praucs'] = np.mean([val for key, val in metrics.items() if key.endswith('_prauc') and val is not None])
        print('Eval on valid set:', hrow['valid_mean_rocaucs'])
        
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
        print(f'Testing on {test_ds_name}:')
        test_subset = test[test['dataset'] == test_ds_name]

        print()
        print('Train classes:')
        for class_name in config.train_classes:
            print(f'{class_name}: {test_subset["fixed_label"].apply(lambda x: class_name in x).sum()}')

        print()
        print('Zeroshot classes:')
        for class_name in config.zeroshot_classes:
            print(f'{class_name}: {test_subset["fixed_label"].apply(lambda x: class_name in x).sum()}')
        
        print('Preparing data:')
        test_ds = lib.dataset.CLIP_ECG_Dataset(test_subset['ecg_file'].values, test_subset['fixed_label'].values, config)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
        print('Testing on train_classes:')
        metrics = lib.training.valid_epoch(net, test_dl, config.train_classes, config) 
        config.test_metrics[test_ds_name] = metrics
        print('Testing on zero_shot classes:')
        metrics = lib.training.valid_epoch(net, test_dl, config.zeroshot_classes, config) 
        config.zero_shot_test_metrics[test_ds_name] = metrics

    config.exp2_metrics_trained = dict()
    config.exp2_metrics_untrained = dict()
    
    for exp2_ds_name in config.test_datasets:
        print(f'Testing on {exp2_ds_name}:')
        df = pd.read_csv(f'docs/{exp2_ds_name}.csv').iloc[:config.limit]

        print()
        print('Train classes:')
        for class_name in config.train_classes:
            print(f'{class_name}: {df["fixed_label"].apply(lambda x: class_name in x).sum()}')

        print()
        print('Zeroshot classes:')
        for class_name in config.zeroshot_classes:
            print(f'{class_name}: {df["fixed_label"].apply(lambda x: class_name in x).sum()}')

        print()
        print('Preparing data:')
        test_ds = lib.dataset.CLIP_ECG_Dataset(df['ecg_file'].values, df['fixed_label'].values, config)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
        print('Testing on train_classes:')
        metrics = lib.training.valid_epoch(net, test_dl, config.train_classes, config) 
        config.exp2_metrics_trained[exp2_ds_name] = metrics
        print('Testing on zero_shot classes:')
        metrics = lib.training.valid_epoch(net, test_dl, config.zeroshot_classes, config) 
        config.exp2_metrics_untrained[exp2_ds_name] = metrics
 
    cfg = {k:v for k, v in config.__dict__.items() if not k.startswith('__')}
    with open(f'{config.logs_path}/{config.name}.cfg', 'w') as fp:
        json.dump(cfg, fp)