
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity


# ==================== initialize functions ====================
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    
    
# ==================== define the model ====================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x

class Cnn14_emb64_Spec(nn.Module):
    def __init__(self, mel_bins=128, classes_num=10):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.fc1 = nn.Linear(2048, 64)
        
        init_bn(self.bn0)
        init_layer(self.fc1)
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.bn0(x)
        x = self.conv_block1(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block5(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block6(x, pool_size=(1,1))
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        embedding = self.fc1(x)
        return embedding


# ==================== Contrastive Loss ====================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.02):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negative):
        # **normalization**
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        # positive similarity 
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        # negative similarity
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # InfoNCE loss: 
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        
        return F.cross_entropy(logits, labels)

# ==================== Triplet Dataset ====================
class TripletDataset(Dataset):
    def __init__(self, info_df, genre_to_tracks, spec_dir):
        self.info_df = info_df
        self.genre_to_tracks = genre_to_tracks
        self.spec_dir = spec_dir
        self.genres = list(genre_to_tracks.keys())
        
    def __len__(self):
        return len(self.info_df)
    
    def __getitem__(self, idx):
        # Anchor
        anchor_row = self.info_df.iloc[idx]
        anchor_id = anchor_row['track_id']
        anchor_genre = anchor_row['track_genre_top']
        
        # Positive: 
        positive_candidates = [tid for tid in self.genre_to_tracks[anchor_genre] if tid != anchor_id]
        if len(positive_candidates) == 0:
            positive_id = anchor_id
        else:
            positive_id = random.choice(positive_candidates)
        
        # Negative: 
        negative_genre = random.choice([g for g in self.genres if g != anchor_genre])
        negative_id = random.choice(self.genre_to_tracks[negative_genre])
        
        # load spectrograms
        try:
            anchor_spec = np.load(os.path.join(self.spec_dir, f"{anchor_id}.npy"))
            positive_spec = np.load(os.path.join(self.spec_dir, f"{positive_id}.npy"))
            negative_spec = np.load(os.path.join(self.spec_dir, f"{negative_id}.npy"))
            
            anchor_spec = pad_spectrogram(anchor_spec)
            positive_spec = pad_spectrogram(positive_spec)
            negative_spec = pad_spectrogram(negative_spec)
            
            return (
                torch.tensor(anchor_spec, dtype=torch.float32),
                torch.tensor(positive_spec, dtype=torch.float32),
                torch.tensor(negative_spec, dtype=torch.float32)
            )
        except Exception as e:
            print(f"Error loading triplet: {e}")
            return None

def collate_triplet(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None, None
    anchors, positives, negatives = zip(*batch)
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def pad_spectrogram(x, target_len=1293):
    cur_len = x.shape[1]
    if cur_len == target_len:
        return x
    elif cur_len < target_len:
        pad_width = target_len - cur_len
        return np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return x[:, :target_len]