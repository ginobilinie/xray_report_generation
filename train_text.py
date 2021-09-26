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
from datasets import MIMIC, NLMCXR
from losses import CELoss
from models import Classifier, TNN
from baselines.transformer.models import LSTM_Attn

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=0)

RELOAD = True # True / False
PHASE = 'TEST' # TRAIN / TEST
DATASET_NAME = 'NLMCXR' # MIMIC / NLMCXR
MODEL_NAME = 'Transformer' # Transformer / LSTM

if DATASET_NAME == 'MIMIC':
    EPOCHS = 20 # Overfitting after 10 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 16 # Fit 1 GPU
    MILESTONES = [10] # Reduce LR by 10 after reaching milestone epochs
    
elif DATASET_NAME == 'NLMCXR':
    EPOCHS = 20 # Start overfitting after 20 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 16 # Fit 1 GPU
    MILESTONES = [10] # Reduce LR by 10 after reaching milestone epochs
    
else:
    raise ValueError('Invalid DATASET_NAME')

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
        
        dataset = MIMIC('/home/hoang/Datasets/MIMIC/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=True, debug_mode=False, train_phase=(PHASE == 'TRAIN'))
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}'.format(MAX_VIEWS, NUM_LABELS)

    elif DATASET_NAME == 'NLMCXR':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114 # (14 diseases + 100 top noun-phrases)
        NUM_CLASSES = 2
        
        dataset = NLMCXR('/home/hoang/Datasets/NLMCXR/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}'.format(MAX_VIEWS, NUM_LABELS)
        
    else:
        raise ValueError('Invalid DATASET_NAME')

    # --- Choose a Model ---
    if MODEL_NAME == 'Transformer':
        LR = 5e-4 
        WD = 1e-2
        NUM_EMBEDS = 256
        NUM_HEADS = 8
        FWD_DIM = 256
        NUM_LAYERS = 1
        DROPOUT = 0.1
            
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=None, tnn=tnn, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
        criterion = CELoss()
            
    elif MODEL_NAME == 'LSTM':
        # Justin et al. hyper-parameters
        LR = 5e-4 
        WD = 1e-2 # Avoid overfitting with L2 regularization
        NUM_EMBEDS = 256
        HIDDEN_SIZE = 128
        DROPOUT = 0.1
        
        model = LSTM_Attn(num_tokens=VOCAB_SIZE, embed_dim=NUM_EMBEDS, hidden_size=HIDDEN_SIZE, num_topics=NUM_LABELS, num_states=NUM_CLASSES, dropout=DROPOUT)
        criterion = CELoss()
    
    else:
        raise ValueError('Invalid MODEL_NAME')

    # --- Main program ---
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))
    
    last_epoch = -1
    best_metric = 0

    checkpoint_path_from = 'checkpoints/{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,COMMENT)
    checkpoint_path_to = 'checkpoints/{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,COMMENT)
    
    if RELOAD:
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))

    if PHASE == 'TRAIN':
        scaler = torch.cuda.amp.GradScaler() # Reduce floating to 16 bits instead of 32 bits
        
        for epoch in range(last_epoch+1, EPOCHS):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, scaler=scaler)
            val_loss, val_outputs, val_targets = test(val_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT)
            test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT)
            scheduler.step()

            val_metric = []
            test_metric = []
            for i in range(NUM_LABELS):
                try:
                    val_metric.append(metrics.roc_auc_score(val_targets.cpu()[...,i], val_outputs.cpu()[...,i,1]))
                except:
                    pass
                try:
                    test_metric.append(metrics.roc_auc_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1]))
                except:
                    pass
            val_metric = np.mean([x for x in val_metric if str(x) != 'nan'])
            test_metric = np.mean([x for x in test_metric if str(x) != 'nan'])
                
            print('Validation Metric: {} | Test Metric: {}'.format(val_metric, test_metric))
            
            if best_metric < val_metric:
                best_metric = val_metric
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_metric, test_metric))
                print('New Best Metric: {}'.format(best_metric))
                print('Saved To:', checkpoint_path_to)
            
    elif PHASE == 'TEST':
        test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT)
        
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
        
        print('Test AUC      : {}'.format(test_auc))
        print('Test F1       : {}'.format(test_f1))
        print('Test Precision: {}'.format(test_prc))
        print('Test Recall   : {}'.format(test_rec))
        print('Test Accuracy : {}'.format(test_acc))
        
    else:
        raise ValueError('Invalid PHASE')