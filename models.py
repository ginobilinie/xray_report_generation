import torch
import torch.nn as nn
import torch.nn.functional as F
        
# --- Transformer Modules ---
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, input, query, pad_mask=None, att_mask=None):
        input = input.permute(1,0,2) # (V,B,E)
        query = query.permute(1,0,2) # (Q,B,E)
        embed, att = self.attention(query, input, input, key_padding_mask=pad_mask, attn_mask=att_mask) # (Q,B,E), (B,Q,V)
        
        embed = self.normalize(embed + query) # (Q,B,E)
        embed = embed.permute(1,0,2) # (B,Q,E)
        return embed, att # (B,Q,E), (B,Q,V)
    
class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, fwd_dim, dropout=0.0):
        super().__init__()
        self.fwd_layer = nn.Sequential(
            nn.Linear(emb_dim, fwd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fwd_dim, emb_dim),
        )
        self.normalize = nn.LayerNorm(emb_dim)

    def forward(self, input):
        output = self.fwd_layer(input) # (B,L,E)
        output = self.normalize(output + input) # (B,L,E)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input, pad_mask=None, att_mask=None):
        emb, att = self.attention(input,input,pad_mask,att_mask)
        emb = self.fwd_layer(emb)
        return emb, att

class TNN(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.1, num_layers=1,
                num_tokens=1, num_posits=1, token_embedding=None, posit_embedding=None):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim) if not token_embedding else token_embedding
        self.posit_embedding = nn.Embedding(num_posits, embed_dim) if not posit_embedding else posit_embedding
        self.transform = nn.ModuleList([TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_index=None, token_embed=None, pad_mask=None, pad_id=-1, att_mask=None):
        if token_index != None:
            if pad_mask == None:
                pad_mask = (token_index == pad_id) # (B,L)
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0],1).to(token_index.device) # (B,L)
            posit_embed = self.posit_embedding(posit_index) # (B,L,E)
            token_embed = self.token_embedding(token_index) # (B,L,E)
            final_embed = self.dropout(token_embed + posit_embed) # (B,L,E)
        elif token_embed != None:
            posit_index = torch.arange(token_embed.shape[1]).unsqueeze(0).repeat(token_embed.shape[0],1).to(token_embed.device) # (B,L)
            posit_embed = self.posit_embedding(posit_index) # (B,L,E)
            final_embed = self.dropout(token_embed + posit_embed) # (B,L,E)
        else:
            raise ValueError('token_index or token_embed must not be None')

        for i in range(len(self.transform)):
            final_embed = self.transform[i](final_embed, pad_mask, att_mask)[0]
            
        return final_embed # (B,L,E)

# --- Convolution Modules ---
class CNN(nn.Module):
    def __init__(self, model, model_type='resnet'):
        super().__init__()
        if 'res' in model_type.lower(): # resnet, resnet-50, resnest-50, ...
            modules = list(model.children())[:-1] # Drop the FC layer
            self.feature = nn.Sequential(*modules[:-1])
            self.average = modules[-1]
        elif 'dense' in model_type.lower(): # densenet, densenet-121, densenet121, ...
            modules = list(model.features.children())[:-1]
            self.feature = nn.Sequential(*modules)
            self.average = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError('Unsupported model_type!')
        
    def forward(self, input):
        wxh_features = self.feature(input) # (B,2048,W,H)
        avg_features = self.average(wxh_features) # (B,2048,1,1)
        avg_features = avg_features.view(avg_features.shape[0], -1) # (B,2048)
        return avg_features, wxh_features

class MVCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        img = input[0] # (B,V,C,W,H)
        pos = input[1] # (B,V)
        B,V,C,W,H = img.shape

        img = img.view(B*V,C,W,H)
        avg, wxh = self.model(img) # (B*V,F), (B*V,F,W,H)
        avg = avg.view(B,V,-1) # (B,V,F)
        wxh = wxh.view(B,V,wxh.shape[-3],wxh.shape[-2],wxh.shape[-1]) # (B,V,F,W,H)
        
        msk = (pos == -1) # (B,V)
        msk_wxh = msk.view(B,V,1,1,1).float() # (B,V,1,1,1) * (B,V,F,C,W,H)
        msk_avg = msk.view(B,V,1).float() # (B,V,1) * (B,V,F)
        wxh = msk_wxh * (-1) + (1-msk_wxh) * wxh
        avg = msk_avg * (-1) + (1-msk_avg) * avg

        wxh_features = wxh.max(dim=1)[0] # (B,F,W,H)
        avg_features = avg.max(dim=1)[0] # (B,F)
        return avg_features, wxh_features

# --- Main Moduldes ---
class Classifier(nn.Module):
    def __init__(self, num_topics, num_states, cnn=None, tnn=None,
                fc_features=2048, embed_dim=128, num_heads=1, dropout=0.1):
        super().__init__()
        
        # For img & txt embedding and feature extraction
        self.cnn = cnn
        self.tnn = tnn
        self.img_features = nn.Linear(fc_features, num_topics * embed_dim) if cnn != None else None
        self.txt_features = MultiheadAttention(embed_dim, num_heads, dropout) if tnn != None else None
        
        # For classification
        self.topic_embedding = nn.Embedding(num_topics, embed_dim)
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        self.attention = MultiheadAttention(embed_dim, num_heads)
        
        # Some constants
        self.num_topics = num_topics
        self.num_states = num_states
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, img=None, txt=None, lbl=None, txt_embed=None, pad_mask=None, pad_id=3, threshold=0.5, get_embed=False, get_txt_att=False):
        # --- Get img and txt features ---
        if img != None: # (B,C,W,H) or ((B,V,C,W,H), (B,V))
            img_features, wxh_features = self.cnn(img) # (B,F), (B,F,W,H)
            img_features = self.dropout(img_features) # (B,F)
            
        if txt != None:
            if pad_id >= 0 and pad_mask == None:
                pad_mask = (txt == pad_id)
            txt_features = self.tnn(token_index=txt, pad_mask=pad_mask) # (B,L,E)
        
        elif txt_embed != None:
            txt_features = self.tnn(token_embed=txt_embed, pad_mask=pad_mask) # (B,L,E)

        # --- Fuse img and txt features ---
        if img != None and (txt != None or txt_embed != None):
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0],1).to(img_features.device) # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0],1).to(img_features.device) # (B,C)
            topic_embed = self.topic_embedding(topic_index) # (B,T,E)
            state_embed = self.state_embedding(state_index) # (B,C,E)
            
            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics, -1) # (B,F) --> (B,T*E) --> (B,T,E)   
            txt_features, txt_attention = self.txt_features(txt_features, topic_embed, pad_mask) # (B,T,E), (B,T,L)
            final_embed = self.normalize(img_features + txt_features) # (B,T,E)
            
        elif img != None:
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0],1).to(img_features.device) # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0],1).to(img_features.device) # (B,C)
            topic_embed = self.topic_embedding(topic_index) # (B,T,E)
            state_embed = self.state_embedding(state_index) # (B,C,E)

            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics, -1) # (B,F) --> (B,T*E) --> (B,T,E)   
            final_embed = img_features # (B,T,E)
            
        elif txt != None or txt_embed != None:
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(txt_features.shape[0],1).to(txt_features.device) # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(txt_features.shape[0],1).to(txt_features.device) # (B,C)
            topic_embed = self.topic_embedding(topic_index) # (B,T,E)
            state_embed = self.state_embedding(state_index) # (B,C,E)

            txt_features, txt_attention = self.txt_features(txt_features, topic_embed, pad_mask) # (B,T,E), (B,T,L)
            final_embed = txt_features # (B,T,E)
            
        else:
            raise ValueError('img and (txt or txt_embed) must not be all none')
        
        # Classifier output
        emb, att = self.attention(state_embed, final_embed) # (B,T,E), (B,T,C)
        
        if lbl != None: # Teacher forcing
            emb = self.state_embedding(lbl) # (B,T,E)
        else:
            emb = self.state_embedding((att[:,:,1] > threshold).long()) # (B,T,E)
            
        if get_embed:
            return att, final_embed + emb # (B,T,C), (B,T,E)
        elif get_txt_att and (txt != None or txt_embed != None):
            return att, txt_attention # (B,T,C), (B,T,L)
        else:
            return att # (B,T,C)

class Generator(nn.Module):
    def __init__(self, num_tokens, num_posits, embed_dim=128, num_heads=1, fwd_dim=256, dropout=0.1, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.posit_embedding = nn.Embedding(num_posits, embed_dim)
        self.transform = nn.ModuleList([TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.num_tokens = num_tokens
        self.num_posits = num_posits
        
    def forward(self, source_embed, token_index=None, source_pad_mask=None, target_pad_mask=None, max_len=300, top_k=1, bos_id=1, pad_id=3, mode='eye'):
        if token_index != None: # --- Training/Testing Phase ---
            # Adding token embedding and posititional embedding.
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0],1).to(token_index.device) # (1,L) --> (B,L)
            posit_embed = self.posit_embedding(posit_index) # (B,L,E)
            token_embed = self.token_embedding(token_index) # (B,L,E)
            target_embed = token_embed + posit_embed # (B,L,E)
            
            # Make embedding, attention mask, pad mask for Transformer Decoder
            final_embed = torch.cat([source_embed,target_embed], dim=1) # (B,T+L,E)
            if source_pad_mask == None:
                source_pad_mask = torch.zeros((source_embed.shape[0],source_embed.shape[1]),device=source_embed.device).bool() # (B,T)
            if target_pad_mask == None:
                target_pad_mask = torch.zeros((target_embed.shape[0],target_embed.shape[1]),device=target_embed.device).bool() # (B,L)
            pad_mask = torch.cat([source_pad_mask,target_pad_mask], dim=1) # (B,T+L)
            att_mask = self.generate_square_subsequent_mask_with_source(source_embed.shape[1], target_embed.shape[1], mode).to(final_embed.device) # (T+L,T+L)

            # Transformer Decoder
            for i in range(len(self.transform)):
                final_embed = self.transform[i](final_embed,pad_mask,att_mask)[0]

            # Make prediction for next tokens
            token_index = torch.arange(self.num_tokens).unsqueeze(0).repeat(token_index.shape[0],1).to(token_index.device) # (1,K) --> (B,K)
            token_embed = self.token_embedding(token_index) # (B,K,E)
            emb, att = self.attention(token_embed,final_embed) # (B,T+L,E), (B,T+L,K)
            
            # Truncate results from source_embed
            emb = emb[:,source_embed.shape[1]:,:] # (B,L,E)
            att = att[:,source_embed.shape[1]:,:] # (B,L,K)
            return att, emb
        
        else: # --- Inference Phase ---
            return self.infer(source_embed, source_pad_mask, max_len, top_k, bos_id, pad_id)

    def infer(self, source_embed, source_pad_mask=None, max_len=100, top_k=1, bos_id=1, pad_id=3):
        outputs = torch.ones((top_k, source_embed.shape[0], 1), dtype=torch.long).to(source_embed.device) * bos_id # (K,B,1) <s>
        scores = torch.zeros((top_k, source_embed.shape[0]), dtype=torch.float32).to(source_embed.device) # (K,B)

        for _ in range(1,max_len):
            possible_outputs = []
            possible_scores = []

            for k in range(top_k):
                output = outputs[k] # (B,L)
                score = scores[k] # (B)
                
                att, emb = self.forward(source_embed, output, source_pad_mask=source_pad_mask, target_pad_mask=(output == pad_id))
                val, idx = torch.topk(att[:,-1,:], top_k) # (B,K)
                log_val = -torch.log(val) # (B,K)
                
                for i in range(top_k):
                    new_output = torch.cat([output, idx[:,i].view(-1,1)], dim=-1) # (B,L+1)
                    new_score = score + log_val[:,i].view(-1) # (B)
                    possible_outputs.append(new_output.unsqueeze(0)) # (1,B,L+1)
                    possible_scores.append(new_score.unsqueeze(0)) # (1,B)
            
            possible_outputs = torch.cat(possible_outputs, dim=0) # (K^2,B,L+1)
            possible_scores = torch.cat(possible_scores, dim=0) # (K^2,B)

            # Pruning the solutions
            val, idx = torch.topk(possible_scores, top_k, dim=0) # (K,B)
            col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0],1) # (K,B)
            outputs = possible_outputs[idx,col_idx] # (K,B,L+1)
            scores = possible_scores[idx,col_idx] # (K,B)

        val, idx = torch.topk(scores, 1, dim=0) # (1,B)
        col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0],1) # (K,B)
        output = outputs[idx,col_idx] # (1,B,L)
        score = scores[idx,col_idx] # (1,B)
        return output.squeeze(0) # (B,L)

    def generate_square_subsequent_mask_with_source(self, src_sz, tgt_sz, mode='eye'):
        mask = self.generate_square_subsequent_mask(src_sz + tgt_sz)
        if mode == 'one': # model can look at surrounding positions of the current index ith
            mask[:src_sz, :src_sz] = self.generate_square_mask(src_sz)
        elif mode == 'eye': # model can only look at the current index ith
            mask[:src_sz, :src_sz] = self.generate_square_identity_mask(src_sz)
        else: # model can look at surrounding positions of the current index ith with some patterns
            raise ValueError('Mode must be "one" or "eye".')
        mask[src_sz:, src_sz:] = self.generate_square_subsequent_mask(tgt_sz)
        return mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_identity_mask(self, sz):
        mask = (torch.eye(sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask 

    def generate_square_mask(self, sz):
        mask = (torch.ones(sz,sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# --- Full Models ---
class ClsGen(nn.Module):
    def __init__(self, classifier, generator, num_topics, embed_dim):
        super().__init__()
        self.classifier = classifier
        self.generator = generator
        self.label_embedding = nn.Embedding(num_topics, embed_dim)

    def forward(self, image, history=None, caption=None, label=None, threshold=0.15, bos_id=1, eos_id=2, pad_id=3, max_len=300, get_emb=False):
        label = label.long() if label != None else label
        img_mlc, img_emb = self.classifier(img=image, txt=history, lbl=label, threshold=threshold, pad_id=pad_id, get_embed=True) # (B,T,C), (B,T,E)
        lbl_idx = torch.arange(img_emb.shape[1]).unsqueeze(0).repeat(img_emb.shape[0],1).to(img_emb.device) # (B,T)
        lbl_emb = self.label_embedding(lbl_idx) # (B,T,E)
        
        if caption != None:
            src_emb = img_emb + lbl_emb
            pad_mask = (caption == pad_id)
            cap_gen, cap_emb = self.generator(source_embed=src_emb, token_index=caption, target_pad_mask=pad_mask) # (B,L,S), (B,L,E)
            if get_emb:
                return cap_gen, img_mlc, cap_emb
            else:
                return cap_gen, img_mlc
        else:
            src_emb = img_emb + lbl_emb
            cap_gen = self.generator(source_embed=src_emb, token_index=caption, max_len=max_len, bos_id=bos_id, pad_id=pad_id) # (B,L,S)
            return cap_gen, img_mlc

class ClsGenInt(nn.Module):
    def __init__(self, clsgen, interpreter, freeze_evaluator=True):
        super().__init__()
        self.clsgen = clsgen
        self.interpreter = interpreter
            
        # Freeze evaluator's paramters
        if freeze_evaluator:
            for param in self.interpreter.parameters():
                param.requires_grad = False

    def forward(self, image, history=None, caption=None, label=None, threshold=0.15, bos_id=1, eos_id=2, pad_id=3, max_len=300):        
        if caption != None:
            pad_mask = (caption == pad_id)
            cap_gen, img_mlc, cap_emb = self.clsgen(image, history, caption, label, threshold, bos_id, eos_id, pad_id, max_len, True)
            cap_mlc = self.interpreter(txt_embed=cap_emb, pad_mask=pad_mask)
            return cap_gen, img_mlc, cap_mlc
        else:
            return self.clsgen(image, history, caption, label, threshold, bos_id, eos_id, pad_id, max_len, False)