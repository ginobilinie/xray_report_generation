# --- Base packages ---
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# --- PyTorch packages ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# --- Project Packages ---
from utils import save, load, train, test
from datasets import NLMCXR, MIMIC
from models import Classifier, TNN
from baselines.transformer.models import LSTM_Attn

# --- Instructions ---
# Step 1: Use train_text.py, train LSTM/Transformer models on the MIMIC-CXR dataset (14 diseases + 100 noun-phrases = 114 labels)
# Step 2: Use extract_label.py, load the NLMCXR dataset and predict labels using the trained LSTM/Transformer models (CheXpert)
# Step 3: Save the predicted labels and load the NLMCXR dataset again with the saved labels
# Step 4: Copy file2label.json to the NLMCXR dataset folder
# Step 5: Use train_text.py, train LSTM/Transformer models on the NLMCXR dataset

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=0)

MODEL_NAME = 'LSTM' # Transformer / LSTM
BATCH_SIZE = 16

if __name__ == "__main__":
    # --- Choose Inputs/Outputs
    if MODEL_NAME == 'Transformer':
        SOURCES = ['caption']
        TARGETS = ['label']
        KW_SRC = ['txt'] # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None
        
    elif MODEL_NAME == 'LSTM':
        SOURCES = ['caption', 'caption_length']
        TARGETS = ['label']
        KW_SRC = ['caption', 'caption_length'] # kwargs of LSTM_Attn
        KW_TGT = None
        KW_OUT = None
        
    else:
        raise ValueError('Invalid MODEL_NAME')
    
    # --- Choose a Dataset ---
    mimic_dataset = MIMIC('/home/hoang/Datasets/MIMIC/', view_pos=['AP','PA','LATERAL'], sources=SOURCES, targets=TARGETS, vocab_file='mimic_unigram_1000.model')    
    dataset = NLMCXR('/home/hoang/Datasets/NLMCXR/', view_pos=['AP','PA','LATERAL'], sources=SOURCES, targets=TARGETS, vocab_file='mimic_unigram_1000.model') 
    # Use the same vocab_file as MIMIC because language models were trained on this.
    
    NUM_LABELS = 114 # (14 diseases + 100 top noun-phrases) <-- MIMIC-CXR
    NUM_CLASSES = 2
    VOCAB_SIZE = len(dataset.vocab)
    POSIT_SIZE = dataset.max_len
    
    # --- Choose a Model ---
    if MODEL_NAME == 'Transformer':
        NUM_EMBEDS = 256
        NUM_HEADS = 8
        FWD_DIM = 256
        NUM_LAYERS = 1
        DROPOUT = 0.1
            
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=None, tnn=tnn, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
            
    elif MODEL_NAME == 'LSTM':
        # Justin et al. hyper-parameters
        NUM_EMBEDS = 256
        HIDDEN_SIZE = 128
        DROPOUT = 0.1
        
        model = LSTM_Attn(num_tokens=VOCAB_SIZE, embed_dim=NUM_EMBEDS, hidden_size=HIDDEN_SIZE, num_topics=NUM_LABELS, num_states=NUM_CLASSES, dropout=DROPOUT)
        
    else:
        raise ValueError('Invalid MODEL_NAME')

    # --- Main program ---
    data_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    model = nn.DataParallel(model).cuda()

    COMMENT = 'MaxView{}_NumLabel{}'.format(2, 114)
    checkpoint_path_from = 'checkpoints/{}_{}_{}.pt'.format('MIMIC',MODEL_NAME,COMMENT)
    last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model)
    print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))
    
    loss, outputs, _ = test(data_loader, model, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT)
    
    # --- Label Extraction ---    
    threshold = 0.5
    label = (outputs[:,:14,1] > threshold).long().cpu().numpy() # Extract only 14 common diseases!
    
    import json
    file_to_label = {}
    for i in range(len(label)):
        file_to_label[dataset.file_list[i]] = label[i].tolist()
    json.dump(file_to_label, open('file2label.json', 'w'))