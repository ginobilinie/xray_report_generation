import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class Transformer(nn.Module):
    def __init__(self, image_encoder, num_tokens, num_posits, fc_features=1024, embed_dim=256, num_heads=8, fwd_dim=4096, dropout=0.1, num_layers_enc=1, num_layers_dec=6, freeze_encoder=True):
        '''
        Reimplemented based on the orginal source code: https://github.com/justinlovelace/coherent-xray-report-generation
        Original paper: https://www.aclweb.org/anthology/2020.findings-emnlp.110.pdf
        '''
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.posit_embedding = nn.Embedding(num_posits, embed_dim)
        self.pixel_embedding = nn.Embedding(64, embed_dim) # last convolution layer has 8x8 pixels = 64 pixels
        
        self.transformer_enc = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(embed_dim,num_heads,fwd_dim,dropout), num_layers=num_layers_enc)
        self.transformer_dec = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(embed_dim,num_heads,fwd_dim,dropout), num_layers=num_layers_dec)
        
        self.fc1 = nn.Linear(fc_features, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_tokens)
        
        self.image_encoder = image_encoder # make sure that image_encoder is a MVCNN model
        if freeze_encoder: # The orginal paper freeze the densenet which is pretrained on ImageNet. Suprisingly, the results were very good
            for param in self.image_encoder.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(dropout)
        self.num_tokens = num_tokens
        self.num_posits = num_posits

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, image, caption = None, bos_id=1, eos_id=2, pad_id=3, max_len=300):
        if caption != None:
            avg_features, wxh_features = self.image_encoder(image) # (B,F), (B,F,W,H)
            wxh_features = wxh_features.view(wxh_features.shape[0], wxh_features.shape[1], -1).permute(0,2,1) # (B,W*H,F)
            wxh_features = self.fc1(wxh_features) # (B,W*H,E)

            pixel = torch.arange(wxh_features.shape[1]).unsqueeze(0).repeat(wxh_features.shape[0],1).to(wxh_features.device)
            pixel_embed = self.pixel_embedding(pixel) # (B,W*H,E)
            img_features = wxh_features + pixel_embed # (B,W*H,E)
            img_features = self.transformer_enc(img_features.transpose(0,1)).transpose(0,1) # (B,W*H,E)
            
            posit = torch.arange(caption.shape[1]).unsqueeze(0).repeat(caption.shape[0],1).to(caption.device) # (1,L) --> (B,L)
            posit_embed = self.posit_embedding(posit) # (B,L,E)
            token_embed = self.token_embedding(caption) # (B,L,E)
            cap_features = token_embed + posit_embed # (B,L,E)
            
            tgt_mask = self.generate_square_subsequent_mask(caption.shape[1]).to(caption.device)
            output = self.transformer_dec(tgt=cap_features.transpose(0,1), 
                                          memory=img_features.transpose(0,1),
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=(caption == pad_id)).transpose(0, 1) # (L,B,E) -> (B,L,E)
            
            preds = self.fc2(self.dropout(output)) # (B,L,S)
            preds = torch.softmax(preds, dim = -1) # (B,L,S)
            return preds # (B,L,S)

        else:
            avg_features, wxh_features = self.image_encoder(image) # (B,F), (B,F,W,H)
            wxh_features = wxh_features.view(wxh_features.shape[0], wxh_features.shape[1], -1).permute(0,2,1) # (B,W*H,F)
            wxh_features = self.fc1(wxh_features) # (B,W*H,E)
            
            pixel = torch.arange(wxh_features.shape[1]).unsqueeze(0).repeat(wxh_features.shape[0],1).to(wxh_features.device)
            pixel_embed = self.pixel_embedding(pixel) # (B,W*H,E)
            img_features = wxh_features + pixel_embed # (B,W*H,E)
            img_features = self.transformer_enc(img_features.transpose(0,1)).transpose(0,1) # (B,W*H,E)
            
            caption = torch.ones((img_features.shape[0],1), dtype=torch.long).to(img_features.device) * bos_id # (B,1)
            for _ in range(max_len):
                posit = torch.arange(caption.shape[1]).unsqueeze(0).repeat(caption.shape[0],1).to(caption.device) # (1,L') --> (B,L')
                posit_embed = self.posit_embedding(posit) # (B,L',E)
                token_embed = self.token_embedding(caption) # (B,L',E)
                cap_features = token_embed + posit_embed # (B,L',E)

                tgt_mask = self.generate_square_subsequent_mask(caption.shape[1]).to(caption.device)
                output = self.transformer_dec(tgt=cap_features.transpose(0,1), 
                                              memory=img_features.transpose(0,1),
                                              tgt_mask=tgt_mask,
                                              tgt_key_padding_mask=(caption == pad_id)).transpose(0, 1) # (L',B,E) -> (B,L',E)
                
                preds = self.fc2(self.dropout(output)) # (B,L',S)
                preds = torch.softmax(preds, dim = -1) # (B,L',S)
                preds = torch.argmax(preds[:,-1,:], dim=-1, keepdim=True) # (B,1)
                caption = torch.cat([caption, preds], dim=-1) # (B,L'+1)
            
            return caption # (B,L')
        
class GumbelTransformer(nn.Module):
    def __init__(self, transformer, diff_chexpert, freeze_chexpert=True):
        '''
        Reimplemented based on the orginal source code: https://github.com/justinlovelace/coherent-xray-report-generation
        Original paper: https://www.aclweb.org/anthology/2020.findings-emnlp.110.pdf
        '''
        super().__init__()
        self.transformer = transformer
        self.diff_chexpert = diff_chexpert
        if freeze_chexpert:
            for param in self.diff_chexpert.parameters():
                param.requires_grad = False

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def apply_chexpert(self, embed, caption_length):
        padding_mask = self.diff_chexpert.generate_pad_mask(embed.shape[0], embed.shape[1], caption_length)
        output, (_, _) = self.diff_chexpert.rnn(embed)
        y_hats = [attn(output, padding_mask) for attn in self.diff_chexpert.attns]
        y_hats = torch.stack(y_hats, dim=1)
        y_hats = torch.softmax(y_hats, dim=-1)
        return y_hats
        
    def forward(self, image, caption = None, caption_length=None, bos_id=1, eos_id=2, pad_id=3, max_len=300, temperature=1, beta=1):
        if caption != None:
            preds = self.transformer(image, caption, bos_id, eos_id, pad_id, max_len) # (B,L,S)
            
            logits = torch.log(preds) # (B,L,S)            
            one_hot_preds = self.gumbel_softmax_sample(logits,temperature,beta) # (B,L,S)
            vocab = torch.arange(self.transformer.num_tokens).unsqueeze(0).repeat(caption.shape[0],1).to(caption.device) # (1,S) --> (B,S)
            vocab_embed = self.transformer.token_embedding(vocab) # (B,S,E)
            preds_embed = one_hot_preds @ vocab_embed # (B,L,S) x (B,S,E) = (B,L,E)
            chexpert_preds = self.apply_chexpert(preds_embed, caption_length) # (B,D,C)
            return preds, chexpert_preds # (B,L,S), (B,D,C)

        else:
            caption = self.transformer(image, caption, bos_id, eos_id, pad_id, max_len) # (B,L')
            return caption # (B,L')
        
    def sample_gumbel(self, shape, device, eps=1e-20):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, beta):
        y = logits + beta * self.sample_gumbel(logits.size(), logits.device)
        return torch.softmax(y / temperature, dim=-1)
    
# --- CheXpert ---
class TanhAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5, num_out=2):
        super(TanhAttention, self).__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, output, mask):
        attn1 = nn.Tanh()(self.attn1(output))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = F.softmax(torch.add(attn2, mask), dim=1)

        h = output.transpose(1, 2).matmul(attn.unsqueeze(2)).squeeze(2)
        y_hat = self.fc(self.dropout(h))
        
        return y_hat

class DotAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5, num_out=2):
        super(DotAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, output, mask):
        attn = (self.attn(output) / (self.hidden_size ** 0.5)).squeeze(-1)
        attn = F.softmax(torch.add(attn, mask), dim=1)

        h = output.transpose(1, 2).matmul(attn.unsqueeze(2)).squeeze(2)
        y_hat = self.fc(self.dropout(h))

        return y_hat
    
class LSTM_Attn(nn.Module):
    def __init__(self, num_tokens, embed_dim, hidden_size, num_topics, num_states, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, embed_dim)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attns = nn.ModuleList([TanhAttention(hidden_size*2, dropout, num_states) for i in range(num_topics)])

    def generate_pad_mask(self, batch_size, max_len, caption_length):
        mask = torch.full((batch_size, max_len), fill_value=float('-inf'), dtype=torch.float, device='cuda')
        for ind, cap_len in enumerate(caption_length):
            mask[ind][:cap_len] = 0
        return mask

    def forward(self, caption, caption_length):
        x = self.embed(caption) # (B,L,E)
        output, (_,_) = self.rnn(x)

        padding_mask = self.generate_pad_mask(caption.shape[0], caption.shape[1], caption_length)

        y_hats = [attn(output, padding_mask) for attn in self.attns]
        y_hats = torch.stack(y_hats, dim=1)
        y_hats = torch.softmax(y_hats, dim=-1)
        return y_hats

class CNN_Attn(nn.Module):
    def __init__(self, embed_weight, emb_dim, filters, kernels, num_classes=14):

        super(CNN_Attn, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)

        self.Ks = kernels

        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K) for K in self.Ks])

        self.attns = nn.ModuleList([DotAttention(filters) for _ in range(num_classes)])

    def generate_pad_mask(self, batch_size, max_len, caption_length):
        total_len = max_len*len(self.Ks)
        for K in self.Ks:
            total_len -= (K-1)
        mask = torch.full((batch_size, total_len), fill_value=float('-inf'), dtype=torch.float, device='cuda')
        for ind1, cap_len in enumerate(caption_length):
            for ind2, K in enumerate(self.Ks):
                mask[ind1][max_len*ind2:cap_len-(K-1)] = 0

        return mask

    def forward(self, encoded_captions, caption_length):
        x = self.embed(encoded_captions).transpose(1, 2)

        batch_size = encoded_captions.size(0)
        max_len = encoded_captions.size(1)
        padding_mask = self.generate_pad_mask(batch_size, max_len, caption_length)

        output = [F.relu(conv(x)).transpose(1, 2) for conv in self.convs]
        output = torch.cat(output, dim=1)


        y_hats = [attn(output, padding_mask) for attn in self.attns]
        y_hats = torch.stack(y_hats, dim=1)

        return y_hats