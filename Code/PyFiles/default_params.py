class CFG:
    seed = 42
    cache_path = 'cache'
    data_path = '/ayb/vol1/datasets/ecg_datasets/physionet.org/files/challenge-2021/1.0.3/training/ptb-xl'
    logs_path = 'results'
    models_path = 'results'
    test_size = 0.20
    valid_size = 0.25
    min_class_count = 200

    batch_size = 256
    num_workers = 12
    ecg_sr = 128
    window = 1280

    
    text_embedding_size = 768
    projection_dim = 256
    dropout = 0.15
    pretrained = True
    text_encoder_model = 'emilyalsentzer/Bio_ClinicalBERT'
    text_tokenizer = 'emilyalsentzer/Bio_ClinicalBERT'   
    temperature = 10.0

    head_lr = 0.001
    image_encoder_lr = 0.001
    device = 'cuda:1'

    epochs = 30
    max_length = 200

    ecg_encoder_channels = (32, 32, 64, 64, 128, 128, 256, 256)
    ecg_encoder_kernels = (7, 7, 5, 5, 3, 3, 3, 3)
    ecg_linear_size = 512
    ecg_embedding_size = 512
    ecg_channels = 12
    
    # debug = False
    # batch_size = 64
    # num_workers = 24
    # head_lr = 0.001
    # image_encoder_lr = 0.001
    # patience = 5
    # factor = 0.8
    # epochs = 100
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # # Image Model
    # model_name = 'resnet18'
    # image_embedding_size = 512

    # # Text Model
    # text_encoder_model = 'emilyalsentzer/Bio_ClinicalBERT'
    # text_tokenizer = 'emilyalsentzer/Bio_ClinicalBERT'
    # text_embedding_size = 768
    # max_length = 200

    # pretrained = True # for both image encoder and text encoder
    # trainable = True # for both image encoder and text encoder
    # temperature = 10.0
    # optimizer = torch.optim.Adam

    # # image size
    # size = 224

    # # for projection head; used for both image and text encoder
    # num_projection_layers = 1
    # projection_dim = 256
    # dropout = 0.0
    # ecg_sr = 128