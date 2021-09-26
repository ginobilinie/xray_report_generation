import torch
import torch.nn as nn
import torch.nn.functional as F

class ST(nn.Module): # Show and Tell
    def __init__(self, image_encoder, num_tokens, fc_features=1024, embed_dim=256, hidden_size=512, dropout=0.1, freeze_encoder=True):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, embed_dim)
        
        self.image_encoder = image_encoder
        if freeze_encoder: # The orginal paper freeze the densenet which is pretrained on ImageNet. Suprisingly, the results were very good
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        
        self.fc1 = nn.Linear(fc_features, embed_dim)
        self.fc2 = nn.Linear(hidden_size, num_tokens)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, image, caption=None, caption_length=None, bos_id=1, eos_id=2, pad_id=3, max_len=300):
        if caption != None:
            avg_features, wxh_features = self.image_encoder(image) # (B,F), (B,F,W,H)
            
            img_features = self.fc1(avg_features) # (B,F) --> (B,E)
            cap_embed = self.embed(caption) # (B,L,E)
            embed = torch.cat([img_features.unsqueeze(1), cap_embed], dim=1) # (B,1+L,E)
            
            output, _ = self.rnn(embed) # (B,1+L,H)
            
            preds = self.fc2(self.dropout(output)) # (B,1+L,S)
            preds = torch.softmax(preds, dim = -1) # (B,1+L,S)
            return preds[:,1:,:] # (B,L,S)
        
        else:
            avg_features, wxh_features = self.image_encoder(image) # (B,F), (B,F,W,H)
            
            img_features = self.fc1(avg_features) # (B,F) --> (B,E)
            caption = torch.ones((img_features.shape[0],1), dtype=torch.long).to(img_features.device) * bos_id # (B,1)
            
            for i in range(max_len):
                cap_embed = self.embed(caption) # (B,L',E)
                embed = torch.cat([img_features.unsqueeze(1), cap_embed], dim=1) # (B,1+L',E)
                
                output, _ = self.rnn(embed) # (B,1+L',H)
                
                preds = self.fc2(self.dropout(output)) # (B,1+L',S)
                preds = torch.softmax(preds, dim = -1) # (B,1+L',S)
                preds = torch.argmax(preds[:,-1,:], dim=-1, keepdim=True) # (B,1)
                caption = torch.cat([caption, preds], dim=-1) # (B,L'+1)
            
            return caption # (B,L')
        
class SAT(nn.Module): # Show, Attend and Tell
    def __init__(self, image_encoder, num_tokens, fc_features=1024, embed_dim=256, hidden_size=512, dropout=0.1, freeze_encoder=True):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, embed_dim)
        
        self.image_encoder = image_encoder
        if freeze_encoder: # The orginal paper freeze the densenet which is pretrained on ImageNet. Suprisingly, the results were very good
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        
        self.fc1 = nn.Linear(fc_features, embed_dim)
        self.fc2 = nn.Linear(hidden_size, num_tokens)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, image, caption=None, caption_length=None, bos_id=1, eos_id=2, pad_id=3, max_len=300):
        if caption != None:
            avg_features, wxh_features = self.image_encoder(image) # (B,F), (B,F,W,H)
            
            img_features = self.fc1(avg_features) # (B,F) --> (B,E)
            cap_embed = self.embed(caption) # (B,L,E)
            embed = torch.cat([img_features.unsqueeze(1), cap_embed], dim=1) # (B,1+L,E)
            
            output, _ = self.rnn(embed) # (B,1+L,H)
            
            preds = self.fc2(self.dropout(output)) # (B,1+L,S)
            preds = torch.softmax(preds, dim = -1) # (B,1+L,S)
            return preds[:,1:,:] # (B,L,S)
        
        else:
            avg_features, wxh_features = self.image_encoder(image) # (B,F), (B,F,W,H)
            
            img_features = self.fc1(avg_features) # (B,F) --> (B,E)
            caption = torch.ones((img_features.shape[0],1), dtype=torch.long).to(img_features.device) * bos_id # (B,1)
            
            for i in range(max_len):
                cap_embed = self.embed(caption) # (B,L',E)
                embed = torch.cat([img_features.unsqueeze(1), cap_embed], dim=1) # (B,1+L',E)
                
                output, _ = self.rnn(embed) # (B,1+L',H)
                
                preds = self.fc2(self.dropout(output)) # (B,1+L',S)
                preds = torch.softmax(preds, dim = -1) # (B,1+L',S)
                preds = torch.argmax(preds[:,-1,:], dim=-1, keepdim=True) # (B,1)
                caption = torch.cat([caption, preds], dim=-1) # (B,L'+1)
            
            return caption # (B,L')