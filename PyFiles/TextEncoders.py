import torch
from transformers import AutoTokenizer, AutoModel
import ast

# Case 1: dx_modality only
class TextEncoderV1:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def encode(self, text_list):
        # Check if text_list is a string representation of a list
        if isinstance(text_list, str):
            text_list = ast.literal_eval(text_list)
        # Convert list of strings to a single string
        text = ', '.join(text_list)
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # Get embeddings from ClinicalBERT model
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state
        # Average the embeddings to get single vector per each input
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings

# Case 2: dx_modality, age, sex
class TextEncoderV2:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def encode(self, series):
        text = f"{series['age']}, {series['sex']}, {series['dx_modality']}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings