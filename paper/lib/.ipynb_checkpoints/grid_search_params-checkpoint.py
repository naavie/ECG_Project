class GRID_SEARCH:
    seed = [42, 43, 44]

    projection_dim = [128, 256, 512]
    dropout = [0.0, 0.15, 0.25]
    temperature = [1.0, 5.0, 10.0]
    head_lr = [0.01, 0.001, 0.0001]
    image_encoder_lr = [0.01, 0.001, 0.0001]

    ecg_encoder_channels = [(32, 32, 64, 64, 128, 256), 
                            (32, 32, 64, 64, 128, 128, 256),
                            (32, 32, 64, 64, 128, 128, 256, 256)]

    ecg_encoder_kernels = [(7, 7, 5, 5, 3, 3),
                           (7, 7, 5, 5, 3, 3, 3),
                           (7, 7, 5, 5, 3, 3, 3, 3)]

    ecg_linear_size = [128, 256, 512]
    ecg_embedding_size = [256, 512, 768]