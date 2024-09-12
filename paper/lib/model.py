import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


_ACTIVATION_DICT = {'relu': nn.ReLU,
                    'tanh': nn.Tanh,
                    'none': nn.Identity,
                    'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2)}


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 act='relu', bn=True, dropout=None,
                 maxpool=None, padding=None, stride=1):

        super().__init__()

        if padding is None or padding == 'same':
            padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=not bn, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.act = _ACTIVATION_DICT[act]()
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        self.maxpool = None if maxpool is None else nn.MaxPool1d(maxpool)

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', bn=True, dropout=None):

        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=not bn)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.act = _ACTIVATION_DICT[act]()
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)

        if self.bn is not None:
            x = self.bn(x)

        x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return x

    
    
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernels, bn=True, dropout=None, maxpool=2, padding=0, stride=None):
        super().__init__()



        num_layers = len(channels)
        if stride is None:
            stride = [1] * num_layers

        self.in_layer = Conv1dBlock(in_channels, channels[0], kernels[0], bn=bn, dropout=dropout, maxpool=maxpool, padding=padding, stride=stride[0])

        conv_layers = list()
        for i in range(1, num_layers):
            conv_layers.append(Conv1dBlock(channels[i - 1], channels[i], kernels[i], bn=bn, dropout=dropout, maxpool=maxpool, padding=padding, stride=stride[i]))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x

    
class ECGEncoder(nn.Module):
    def __init__(self, config):
        
        super().__init__()

        channels = config.ecg_encoder_channels
        kernels = config.ecg_encoder_kernels
        linear = config.ecg_linear_size
        output = config.ecg_embedding_size
        window = config.window
        in_channels = config.ecg_channels
        
        self.conv_encoder = ConvEncoder(in_channels, channels,  kernels, bn=True)
        
        with torch.no_grad():
            inpt = torch.zeros((1, in_channels, window), dtype=torch.float32)
            outpt = self.conv_encoder(inpt)
            output_window = outpt.shape[2]
            
        self.flatten = nn.Flatten()
        self.conv_to_linear = nn.Linear(output_window * channels[-1], linear)
        self.act = nn.ReLU()
        self.out_layer = nn.Linear(linear, output)
        
    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.flatten(x)
        x = self.conv_to_linear(x)
        x = self.act(x)
        x = self.out_layer(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        if config.pretrained:
            self.model = AutoModel.from_pretrained(config.text_encoder_model)
        else:
            self.model = AutoModel.from_config(config.text_encoder_model)
            
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer)

        for p in self.model.parameters():
            p.requires_grad = False  # Set requires_grad to False for all parameters

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, texts):
        input_ids, attention_mask = self.tokenize_texts(texts)
        embeddinbgs = self.inputs_to_embeddings(input_ids, attention_mask)
        return embeddinbgs
    
    def tokenize_texts(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].detach().to(self.config.device)
        attention_mask = inputs['attention_mask'].detach().to(self.config.device)      
        return input_ids, attention_mask

    def inputs_to_embeddings(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :].detach()

class ProjectionHead(nn.Module):
    def __init__(self, config, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, config.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(config.projection_dim, config.projection_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        image_embedding=config.ecg_embedding_size
        text_embedding=config.text_embedding_size
        
        self.image_encoder = ECGEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.image_projection = ProjectionHead(config, embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(config, embedding_dim=text_embedding)
        self.temperature = config.temperature

    def forward(self, batch):
        image_embeddings = self.image_to_embeddings(batch['image'])
        text_embeddings = self.text_to_embeddings(batch['caption'])

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean(), image_embeddings, text_embeddings
    
    def text_to_embeddings(self, texts):
        text_features = self.text_encoder(texts)
        text_embeddings = self.text_projection(text_features)
        return text_embeddings
    
    def image_to_embeddings(self, images):
        image_features = self.image_encoder(images)
        image_embeddings = self.image_projection(image_features)
        return image_embeddings

    
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
    

