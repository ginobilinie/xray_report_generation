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
from datasets import MIMIC, NLMCXR, TextDataset
from models import Classifier, TNN
from baselines.transformer.models import LSTM_Attn

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=0)

DATASET_NAME = 'MIMIC' # MIMIC / NLMCXR
MODEL_NAME = 'LSTM' # Transformer / LSTM
BATCH_SIZE = 32

TEXT_FILE = 'outputs/{}_ClsGen_DenseNet121_MaxView2_NumLabel114_NoHistory_Hyp.txt'.format(DATASET_NAME)
LABEL_FILE = 'outputs/{}_ClsGen_DenseNet121_MaxView2_NumLabel114_NoHistory_Lbl.txt'.format(DATASET_NAME)

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
    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2        
        NUM_LABELS = 114 # (14 diseases + 100 top noun-phrases)
        NUM_CLASSES = 2

        dataset = TextDataset(text_file=TEXT_FILE, label_file=LABEL_FILE, sources=SOURCES, targets=TARGETS,
                              vocab_file='/home/hoang/Datasets/MIMIC/mimic_unigram_1000.model', max_len=1000)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}'.format(MAX_VIEWS, NUM_LABELS)

    elif DATASET_NAME == 'NLMCXR':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2        
        NUM_LABELS = 114 # (14 diseases + 100 top noun-phrases)
        NUM_CLASSES = 2
        
        dataset = TextDataset(text_file=TEXT_FILE, label_file=LABEL_FILE, sources=SOURCES, targets=TARGETS,
                              vocab_file='/home/hoang/Datasets/NLMCXR/nlmcxr_unigram_1000.model', max_len=1000)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}'.format(MAX_VIEWS, NUM_LABELS)
        
    else:
        raise ValueError('Invalid DATASET_NAME')
    
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

    checkpoint_path_from = 'checkpoints/{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,COMMENT)
    last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model)
    print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))
    
    test_loss, test_outputs, test_targets = test(data_loader, model, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT)
    
    # --- Evaluation ---
    test_auc = []
    test_f1 = []
    test_prc = []
    test_rec = []
    test_acc = []
    
    threshold = 0.5
    NUM_LABELS = 14
    for i in range(NUM_LABELS):
        try:
            test_auc.append(metrics.roc_auc_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1]))
            test_f1.append(metrics.f1_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
            test_prc.append(metrics.precision_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
            test_rec.append(metrics.recall_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
            test_acc.append(metrics.accuracy_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
            
        except:
            print('An error occurs for label', i)
            
    test_auc = np.mean([x for x in test_auc if str(x) != 'nan'])
    test_f1 = np.mean([x for x in test_f1 if str(x) != 'nan'])
    test_prc = np.mean([x for x in test_prc if str(x) != 'nan'])
    test_rec = np.mean([x for x in test_rec if str(x) != 'nan'])
    test_acc = np.mean([x for x in test_acc if str(x) != 'nan'])
    
    print('Accuracy       : {}'.format(test_acc))
    print('Macro AUC      : {}'.format(test_auc))
    print('Macro F1       : {}'.format(test_f1))
    print('Macro Precision: {}'.format(test_prc))
    print('Macro Recall   : {}'.format(test_rec))
    print('Micro AUC      : {}'.format(metrics.roc_auc_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1], average='micro')))
    print('Micro F1       : {}'.format(metrics.f1_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
    print('Micro Precision: {}'.format(metrics.precision_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
    print('Micro Recall   : {}'.format(metrics.recall_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))