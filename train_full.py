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

# --- Helper Packages ---
from tqdm import tqdm

# --- Project Packages ---
from utils import save, load, train, test, data_to_device, data_concatenate
from datasets import NIHCXR, MIMIC, NLMCXR
from losses import CELoss, CELossTotal, CELossTotalEval, CELossTransfer, CELossShift
from models import CNN, MVCNN, TNN, Classifier, Generator, ClsGen, ClsGenInt
from baselines.transformer.models import LSTM_Attn, Transformer, GumbelTransformer
from baselines.rnn.models import ST

# --- Helper Functions ---
def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    return threshold[ix]

def infer(data_loader, model, device='cpu', threshold=None):
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)

            # Use single input if there is no clinical history
            if threshold != None:
                output = model(image=source[0], history=source[3], threshold=threshold)
                # output = model(image=source[0], threshold=threshold)
                # output = model(image=source[0], history=source[3], label=source[2])
                # output = model(image=source[0], label=source[2])
            else:
                # output = model(source[0], source[1])
                output = model(source[0])
                
            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)
    
    return outputs, targets

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=123)

RELOAD = True # True / False
PHASE = 'INFER' # TRAIN / TEST / INFER
DATASET_NAME = 'MIMIC' # NIHCXR / NLMCXR / MIMIC 
BACKBONE_NAME = 'DenseNet121' # ResNeSt50 / ResNet50 / DenseNet121
MODEL_NAME = 'ClsGenInt' # ClsGen / ClsGenInt / VisualTransformer / GumbelTransformer

if DATASET_NAME == 'MIMIC':
    EPOCHS = 50 # Start overfitting after 20 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64 # 128 # Fit 4 GPUs
    MILESTONES = [25] # Reduce LR by 10 after reaching milestone epochs
    
elif DATASET_NAME == 'NLMCXR':
    EPOCHS = 50 # Start overfitting after 20 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64 # Fit 4 GPUs
    MILESTONES = [25] # Reduce LR by 10 after reaching milestone epochs
    
else:
    raise ValueError('Invalid DATASET_NAME')

if __name__ == "__main__":
    # --- Choose Inputs/Outputs
    if MODEL_NAME in ['ClsGen', 'ClsGenInt']:
        SOURCES = ['image','caption','label','history']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','label','history']
        KW_TGT = None
        KW_OUT = None
                
    elif MODEL_NAME == 'VisualTransformer':
        SOURCES = ['image','caption']
        TARGETS = ['caption']# ,'label']
        KW_SRC = ['image','caption'] # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None
        
    elif MODEL_NAME == 'GumbelTransformer':
        SOURCES = ['image','caption','caption_length']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','caption_length'] # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')
        
    # --- Choose a Dataset ---
    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2
        
        dataset = MIMIC('/home/hoang/Datasets/MIMIC/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=True, debug_mode=False, train_phase=(PHASE == 'TRAIN'))
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
            
    elif DATASET_NAME == 'NLMCXR':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2

        dataset = NLMCXR('/home/hoang/Datasets/NLMCXR/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
        
    else:
        raise ValueError('Invalid DATASET_NAME')

    # --- Choose a Backbone --- 
    if BACKBONE_NAME == 'ResNeSt50':
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        backbone = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        FC_FEATURES = 2048
        
    elif BACKBONE_NAME == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        FC_FEATURES = 2048
        
    elif BACKBONE_NAME == 'DenseNet121':
        backbone = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)
        FC_FEATURES = 1024
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')

    # --- Choose a Model ---
    if MODEL_NAME == 'ClsGen':
        LR = 3e-4 # Fastest LR
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        FWD_DIM = 256
        
        NUM_HEADS = 8
        NUM_LAYERS = 1
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        
        # Not enough memory to run 8 heads and 12 layers, instead 1 head is enough
        NUM_HEADS = 1
        NUM_LAYERS = 12
        
        cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
        gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)
        
        model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
        criterion = CELossTotal(ignore_index=3)
        
    elif MODEL_NAME == 'ClsGenInt':
        LR = 3e-5 # Slower LR to fine-tune the model (Open-I)
        # LR = 3e-6 # Slower LR to fine-tune the model (MIMIC)
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        FWD_DIM = 256
        
        NUM_HEADS = 8
        NUM_LAYERS = 1
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        
        # Not enough memory to run 8 heads and 12 layers, instead 1 head is enough
        NUM_HEADS = 1
        NUM_LAYERS = 12
        
        cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
        gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)
        
        clsgen_model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
        clsgen_model = nn.DataParallel(clsgen_model).cuda()
        
        if not RELOAD:
            checkpoint_path_from = 'checkpoints/{}_ClsGen_{}_{}.pt'.format(DATASET_NAME, BACKBONE_NAME, COMMENT)
            last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, clsgen_model)
            print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))
        
        # Initialize the Interpreter module
        NUM_HEADS = 8
        NUM_LAYERS = 1
        
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        int_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=None, tnn=tnn, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
        int_model = nn.DataParallel(int_model).cuda()
        
        if not RELOAD:
            checkpoint_path_from = 'checkpoints/{}_Transformer_MaxView2_NumLabel{}.pt'.format(DATASET_NAME, NUM_LABELS)
            last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, int_model)
            print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))
        
        model = ClsGenInt(clsgen_model.module.cpu(), int_model.module.cpu(), freeze_evaluator=True)
        criterion = CELossTotalEval(ignore_index=3)
        
    elif MODEL_NAME == 'VisualTransformer':
        # Clinical Coherent X-ray Report (Justin et. al.) - No Fine-tune
        LR = 5e-5
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        NUM_HEADS = 8
        FWD_DIM = 4096
        NUM_LAYERS_ENC = 1
        NUM_LAYERS_DEC = 6
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        model = Transformer(image_encoder=cnn, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, 
                            fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, 
                            dropout=DROPOUT, num_layers_enc=NUM_LAYERS_ENC, num_layers_dec=NUM_LAYERS_DEC, freeze_encoder=True)
        criterion = CELossShift(ignore_index=3)
        
    elif MODEL_NAME == 'GumbelTransformer':
        # Clinical Coherent X-ray Report (Justin et. al.)        
        LR = 5e-5
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        NUM_HEADS = 8
        FWD_DIM = 4096
        NUM_LAYERS_ENC = 1
        NUM_LAYERS_DEC = 6
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        transformer = Transformer(image_encoder=cnn, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, 
                                  fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, 
                                  dropout=DROPOUT, num_layers_enc=NUM_LAYERS_ENC, num_layers_dec=NUM_LAYERS_DEC, freeze_encoder=True)
        transformer = nn.DataParallel(transformer).cuda()
        pretrained_from = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,'VisualTransformer',BACKBONE_NAME,COMMENT)
        last_epoch, (best_metric, test_metric) = load(pretrained_from, transformer)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(pretrained_from, last_epoch, best_metric, test_metric))
        
        NUM_EMBEDS = 256
        HIDDEN_SIZE = 128
        
        pretrained_from = 'checkpoints/{}_{}_{}_NumLabel{}.pt'.format(DATASET_NAME,'LSTM','MaxView2',NUM_LABELS)
        diff_chexpert = LSTM_Attn(num_tokens=VOCAB_SIZE, embed_dim=NUM_EMBEDS, hidden_size=HIDDEN_SIZE, num_topics=NUM_LABELS, num_states=NUM_CLASSES, dropout=DROPOUT)
        diff_chexpert = nn.DataParallel(diff_chexpert).cuda()
        last_epoch, (best_metric, test_metric) = load(pretrained_from, diff_chexpert)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(pretrained_from, last_epoch, best_metric, test_metric))
        
        model = GumbelTransformer(transformer.module.cpu(), diff_chexpert.module.cpu())
        criterion = CELossTotal(ignore_index=3)
        
    elif MODEL_NAME == 'ST':
        KW_SRC = ['image', 'caption', 'caption_length']
        
        LR = 5e-5
        NUM_EMBEDS = 256
        HIDDEN_SIZE = 128
        DROPOUT = 0.1
        
        model = ST(cnn, num_tokens=VOCAB_SIZE, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, freeze_encoder=True)
        criterion = CELossShift(ignore_index=3)
        
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
    best_metric = 1e9

    checkpoint_path_from = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    checkpoint_path_to = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    
    if RELOAD:
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler) # Reload
        # last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model) # Fine-tune
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))

    if PHASE == 'TRAIN':
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(last_epoch+1, EPOCHS):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, scaler=scaler)
            val_loss = test(val_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            test_loss = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            
            scheduler.step()
            
            if best_metric > val_loss:
                best_metric = val_loss
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print('New Best Metric: {}'.format(best_metric)) 
                print('Saved To:', checkpoint_path_to)
    
    elif PHASE == 'TEST':
        # Output the file list for inspection
        out_file_img = open('outputs/{}_{}_{}_{}_Img.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        for i in range(len(test_data.idx_pidsid)):
            out_file_img.write(test_data.idx_pidsid[i][0] + ' ' + test_data.idx_pidsid[i][1] + '\n')
            
        # test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, select_outputs=[1])
        
        # test_auc = []
        # test_f1 = []
        # test_prc = []
        # test_rec = []
        # test_acc = []
        
        # threshold = 0.25
        # NUM_LABELS = 14
        # for i in range(NUM_LABELS):
        #     try:
        #         test_auc.append(metrics.roc_auc_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1]))
        #         test_f1.append(metrics.f1_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_prc.append(metrics.precision_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_rec.append(metrics.recall_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_acc.append(metrics.accuracy_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
                
        #     except:
        #         print('An error occurs for label', i)
                
        # test_auc = np.mean([x for x in test_auc if str(x) != 'nan'])
        # test_f1 = np.mean([x for x in test_f1 if str(x) != 'nan'])
        # test_prc = np.mean([x for x in test_prc if str(x) != 'nan'])
        # test_rec = np.mean([x for x in test_rec if str(x) != 'nan'])
        # test_acc = np.mean([x for x in test_acc if str(x) != 'nan'])
        
        # print('Accuracy       : {}'.format(test_acc))
        # print('Macro AUC      : {}'.format(test_auc))
        # print('Macro F1       : {}'.format(test_f1))
        # print('Macro Precision: {}'.format(test_prc))
        # print('Macro Recall   : {}'.format(test_rec))
        # print('Micro AUC      : {}'.format(metrics.roc_auc_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1], average='micro')))
        # print('Micro F1       : {}'.format(metrics.f1_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        # print('Micro Precision: {}'.format(metrics.precision_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        # print('Micro Recall   : {}'.format(metrics.recall_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        
    elif PHASE == 'INFER':
        txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.25)
        gen_outputs = txt_test_outputs[0]
        gen_targets = txt_test_targets[0]
        
        out_file_ref = open('outputs/x_{}_{}_{}_{}_Ref.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_hyp = open('outputs/x_{}_{}_{}_{}_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_lbl = open('outputs/x_{}_{}_{}_{}_Lbl.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        
        for i in range(len(gen_outputs)):
            candidate = ''
            for j in range(len(gen_outputs[i])):
                tok = dataset.vocab.id_to_piece(int(gen_outputs[i,j]))
                if tok == '</s>':
                    break # Manually stop generating token after </s> is reached
                elif tok == '<s>':
                    continue
                elif tok == '▁': # space
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' '
                elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' ' + tok + ' ' 
                    else:
                        candidate += tok + ' '
                else: # letter
                    candidate += tok       
            out_file_hyp.write(candidate + '\n')
            
            reference = ''
            for j in range(len(gen_targets[i])):
                tok = dataset.vocab.id_to_piece(int(gen_targets[i,j]))
                if tok == '</s>':
                    break
                elif tok == '<s>':
                    continue
                elif tok == '▁': # space
                    if len(reference) and reference[-1] != ' ':
                        reference += ' '
                elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
                    if len(reference) and reference[-1] != ' ':
                        reference += ' ' + tok + ' ' 
                    else:
                        reference += tok + ' '
                else: # letter
                    reference += tok    
            out_file_ref.write(reference + '\n')

        for i in tqdm(range(len(test_data))):
            target = test_data[i][1] # caption, label
            out_file_lbl.write(' '.join(map(str,target[1])) + '\n')
                
    else:
        raise ValueError('Invalid PHASE')