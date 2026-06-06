#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import shap 

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

from IPython.display import display

from captum.attr import LayerIntegratedGradients, IntegratedGradients
from functools import partial
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import cross_fit_m_estimate_oof, apply_m_estimate_map, ordencode

import os, time, random
from tqdm import tqdm
import gc
import ctypes
from dataclasses import dataclass, asdict
import json


def clean_memory():
    """Enhanced memory cleanup for both RAM and VRAM"""
    # Move any remaining tensors to CPU if needed
    # (only if you have model references you want to preserve)
    
    # Synchronize CUDA operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Collect garbage
    gc.collect()
    
    # Free GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Trim RAM (Linux-specific)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass  # Silently fail on non-Linux systems
    
    # Optional: Reset peak memory stats for monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

#del model
clean_memory()


def seed_everything(seed=1337):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Changed from manual_seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Added this line
    
seed_everything()

# Create log directory
LOGDIR = Path(__file__).parent / 'logs'
LOGDIR.mkdir(exist_ok=True)
LOGFILE = LOGDIR / 'ml.log'

# Set up logging, for file and console
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"📝 Logging to {LOGFILE}")

#%%


@dataclass
class Config:
    # Training
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 512
    patience: int = 15
    weight_decay: float = 1e-4
    random_state: int = 1337
    
    # Architecture
    hidden_dims: list = [512, 512, 256, 256]  # [256, 128, 128]
    dropout_rates: list = [0.1, 0.2, 0.2, 0.3]  # [0.2, 0.2, 0.1]
    emb_dropout: float = 0.1
    
    # CV
    k_folds: int = 5
    nsplit_tem: int = 5
    
    # Feature engineering
    te_m: float = 5.0
    
cfg = Config()

#%%

def load_data():
    try:
        df_train = pd.read_csv('data/raw/train.csv').drop(columns=['id'])
        df_test = pd.read_csv('data/raw/test.csv').drop(columns=['id'])

    except Exception as e:
        logger.error(f'Failed to load data: {e}')
        raise
    
    return df_train, df_test

df_train, df_test = load_data()

display(df_train)
print(df_train.info())

target = ['diagnosed_diabetes']
cats = df_train.select_dtypes(include=['object']).columns.difference(target).tolist()
print(cats)
print(len(cats))
nums = df_train.select_dtypes(exclude=['object']).columns.difference(target).tolist()
print(nums)
print(len(nums))

cats = [ 'alcohol_consumption_per_week', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history','gender', 'ethnicity', 'education_level',
       'income_level', 'smoking_status', 'employment_status']
nums = ['age', 'physical_activity_minutes_per_week', 'diet_score','sleep_hours_per_day',
        'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 
        'diastolic_bp', 'heart_rate','cholesterol_total','hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides']


#%% Ordinal encoding
def ordinal_encoding(df1, df2):
    train_len = len(df1)
    df = pd.concat([df1, df2], axis=0)
    
    for cat in cats:
        df[cat], _ = df[cat].factorize()     

    df1 = df.iloc[:train_len, :].copy()
    df2 = df.iloc[train_len:, :].copy()
    df2 = df2.drop(columns=target)
    return df1, df2

df_train, df_test = ordinal_encoding(df_train, df_test)
df_train[cats] = df_train[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
#df_train[nums] = df_train[nums].astype(np.float32)
#df_test[nums] = df_test[nums].astype(np.float32)
print(df_train.shape)
print(df_test.shape)


#%%
# Storage
te_maps = {}
te_prior = {}
te_train_feats = []

# Encode training data (ONE column at a time)
print("Encoding training data...")
for col in nums:
    print(f"  {col}...", end=" ")
    
    oof, full_map, prior = cross_fit_m_estimate_oof(
        df=df_train,                      # Full DataFrame
        y=df_train[target[0]].values,     # Target as numpy array
        col=col,                          # Single column name
        n_splits=cfg.nsplit_tem,
        m=cfg.te_m,
        seed=cfg.random_state
    )
    
    te_maps[col] = full_map
    te_prior[col] = prior
    te_train_feats.append(oof.reshape(-1, 1))
    
    print(f"prior={prior:.4f}, mean={oof.mean():.4f}")


# Apply to test
te_test_feats = []
for col in nums:
    te_test = apply_m_estimate_map(
        df=df_test,
        col=col,
        full_map=te_maps[col],
        prior=te_prior[col],
        m=5.0
    )
    te_test_feats.append(te_test.reshape(-1, 1))

Xte_te = np.concatenate(te_test_feats, axis=1)
print(f"Test TE shape: {Xte_te.shape}")


# Your column lists
nums_te = []
for cname in nums:
    new_col = f'{cname}_te'
    nums_te.append(new_col)


# Adding new columns to the DataFrame
for i,cname in enumerate(nums_te):
    df_train[cname] = te_train_feats[i]
    df_test[cname] = te_test_feats[i]

display(df_train)
print(df_train.shape, df_test.shape)  

#%%Torch classes and model
# Fast Loader is for batches.

class FastDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:, cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:, num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.dfy)
    
    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, len(self))
        cat_val = self.cat_features[start_idx:end_idx]
        num_val = self.num_features[start_idx:end_idx]
        y = self.dfy[start_idx:end_idx]
        return num_val, cat_val, y


class FastDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False, device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.dataset_len = len(dataset)
        
        # Calculate number of batches
        self.n_batches = (self.dataset_len + batch_size - 1) // batch_size
        
    def __iter__(self):
        if self.shuffle:
            # Create shuffled indices
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= self.n_batches:
            raise StopIteration
        
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.dataset_len)
        
        if self.shuffle:
            # Get batch indices from shuffled order
            batch_indices = self.indices[start_idx:end_idx]
            num_val = self.dataset.num_features[batch_indices].to(self.device)
            cat_val = self.dataset.cat_features[batch_indices].to(self.device)
            y = self.dataset.dfy[batch_indices].to(self.device)
        else:
            # Direct slicing for sequential access
            num_val = self.dataset.num_features[start_idx:end_idx].to(self.device)
            cat_val = self.dataset.cat_features[start_idx:end_idx].to(self.device)
            y = self.dataset.dfy[start_idx:end_idx].to(self.device)
        
        self.batch_idx += 1
        return num_val, cat_val, y

    def __len__(self):
        return self.n_batches


class StdDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self, idx):
        cat = self.cat_features[idx]
        num = self.num_features[idx]
        y = self.dfy[idx]
        return [num, cat, y]


#%% 


class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        #print(self.patience)
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            #self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'models/best.pt')
            #print('first best score')
        elif score >= self.best_score:
            self.counter += 1
            #print('counter', self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'models/best.pt')
            self.counter = 0
            #print('counter reset')
            
    def load_best_model(self, model):
#        model.load_state_dict(self.best_model_state)           
        model_data = torch.load('models/best.pt', weights_only=False)
        model.load_state_dict(model_data)       

#%% Make model


class Model(nn.Module):
    def __init__(self, meta_data, emb_dropout, fc_in_out, dropout_perc, d_out=1):
        super().__init__()
        n_num_cols = meta_data['num_nums']
        emb_sizes = meta_data['emb_sizes']
        # Get embedding
        self.embedding_d = nn.ModuleList([nn.Embedding(car,siz) for car,siz in emb_sizes])
        for emb in self.embedding_d:
            emb.weight.data.uniform_(-0.01, 0.01)
            #nn.init.kaiming_normal_(emb.weight.data)
            
        # Embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        # Calculate in_features to linear layer
        emb_vector_sum = sum([e.embedding_dim for e in self.embedding_d])
        # Add in_feature to list
        linear_szs = [emb_vector_sum + n_num_cols] + fc_in_out
        
        # Initialize fc layers
        self.fc_layers = nn.ModuleList([nn.Linear(linear_szs[i],linear_szs[i+1])
                                        for i in range(len(linear_szs) - 1)])
        # Output layer
        self.out = nn.Linear(linear_szs[-1],d_out)
        # Initialize Batch Norm 
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(s) for s in linear_szs[1:]])
        # Batch for num in
        self.batchnorm_num = nn.BatchNorm1d(n_num_cols)
        # Dropout
        self.dropout = nn.ModuleList([nn.Dropout(p) for p in dropout_perc])
    
    
    def forward(self, num_fields, cat_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embedding_d)]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)
        # Dropout for embeddings
        x1 = self.emb_dropout(x1)
        
        # Input normalization for cont fields
        #x2 = self.batchnorm_num(num_fields)
        # Concat inputs
        x1 = torch.cat([x1, num_fields], 1)
        
        for fc, bn, drop in zip(self.fc_layers, self.batchnorm, self.dropout):
            x1 = F.gelu(fc(x1))
            x1 = bn(x1)
            x1 = drop(x1)
        
        x1 = self.out(x1)
        out = x1 #F.sigmoid(x1) #sigmoid as we use BCELoss
        return out


#%%

def get_postsplit_meta(Xtrain, meta_data):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''
    embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[meta_data['CATS']].items()}
    emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]
    meta_data['emb_sizes'] = emb_sizes
    return meta_data


def train(model, loader, optimizer, criterion, DEVICE):
    running_loss = 0.0
    model.train()
    for data in tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        output = model.forward(in1, in2)
        loss = criterion(output, label.float()) #torch.flatten(output)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    training_loss = running_loss/len(loader)
    return training_loss


def valid(model, loader, criterion, DEVICE):
    y_prediction = []
    y_true = []
    running_loss = 0.0
    model.eval()
    for data in loader: #tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        
        output = model.forward(in1, in2)
        loss = criterion(output, label.float())
        running_loss += loss.item()
        
        y_prediction.append(output.detach().cpu().tolist())
        y_true.append(label.detach().cpu().tolist())
    
    
    # Flatten prediction and labels    
    y_true1 = np.array([v for lst in y_true for v in lst])
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])

    validation_loss = running_loss/len(loader)
    validation_rocauc = roc_auc_score( y_true1, y_prediction1 )
    validation_prauc = average_precision_score(y_true1, y_prediction1)
    bin_prediction = (y_prediction1.flatten() >= 0.5).astype(int)
    validation_f1 = f1_score(y_true1, bin_prediction)
    validation_metric = (validation_rocauc, validation_prauc, validation_f1)
    return validation_loss, validation_metric, y_prediction1 


def test_predictions(model, loader, DEVICE):
    y_prediction = []
    model.eval()
    for data in tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        output = model.forward(in1, in2)
        
        y_prediction.append(output.detach().cpu().tolist())
        
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])

    return y_prediction1

#%%

def compute_captum_importance(model, X_val, nums, cats, m, s,
                               n_samples=200, seed=42):
    """
    Compute feature importance via Captum IG (numericals) + LayerIG (categoricals).
    
    Args:
        model : Trained PyTorch model in eval mode
        X_val : pd.DataFrame, validation set features for this fold
        nums  : list of numerical column names
        cats  : list of categorical column names
        m     : pd.Series, numerical feature means (training set)
        s     : pd.Series, numerical feature stds (training set)
        n_samples : int, number of background samples
        seed  : int, random seed for reproducibility
    
    Returns:
        importance : pd.Series, feature names -> mean absolute attribution
        fig : matplotlib Figure object (optional plot)
    """
    # --- Prepare data ---
    model.eval()
    model.to('cpu')
    
    df_X = X_val.copy()
    df_X[nums] = (df_X[nums] - m) / s
    
    sample = df_X.sample(n=min(n_samples, len(df_X)), random_state=seed)
    
    num_t = torch.from_numpy(sample[nums].to_numpy(dtype=np.float32))
    cat_t = torch.from_numpy(sample[cats].to_numpy(dtype=np.int64))
    
    num_baseline = torch.zeros_like(num_t)
    cat_baseline = torch.zeros_like(cat_t)
    
    # --- 1) Numericals: IG ---
    def forward_num_only(num_input, cat_input):
        return model(num_input, cat_input)
    
    ig_num = IntegratedGradients(forward_num_only)
    attrs_num = ig_num.attribute(
        inputs=num_t,
        baselines=num_baseline,
        additional_forward_args=(cat_t,),
        n_steps=50,
        internal_batch_size=256
    )  # shape: (n_samples, n_num)
    
    # --- 2) Categoricals: LayerIG per embedding ---
    cat_attrs_list = []
    for i, emb_layer in enumerate(model.embedding_d):
        
        def forward_cat_i(cat_col_i, num_input, full_cat, idx=i):
            cat_mod = full_cat.clone()
            cat_mod[:, idx] = cat_col_i.squeeze(1).long()
            return model(num_input, cat_mod)
        
        lig = LayerIntegratedGradients(
            lambda x, n=num_t, c=cat_t, idx=i: forward_cat_i(x, n, c, idx),
            emb_layer
        )
        attr_i = lig.attribute(
            inputs=cat_t[:, i].unsqueeze(1),
            baselines=cat_baseline[:, i].unsqueeze(1),
            n_steps=50,
            internal_batch_size=256
        )  # shape: (n_samples, 1, emb_dim)
        
        # Sum over embedding dim, then mean over samples → scalar per feature
        cat_attrs_list.append(attr_i.sum(dim=-1).abs().mean().item())
    
    # --- 3) Assemble importance ---
    num_imp = attrs_num.abs().mean(dim=0).detach().numpy()
    
    importance = pd.Series(
        np.concatenate([num_imp, cat_attrs_list]),
        index=nums + cats
    )
    
    return importance


def log_fold_attributions(fold_idx, model, X_val, nums, cats,
                          m, s, DEVICE, logdir='runs/nn_base',
                          n_samples=200, seed=42):
    """Compute Captum importance and log to TensorBoard for a single fold."""
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt

    writer = SummaryWriter(f'{logdir}/fold{fold_idx}')

    # --- Captum: importances ---
    model.eval()
    model.to('cpu')

    df = X_val.loc[X_val.index].copy()
    df[nums] = (df[nums] - m) / s
    sample = df.sample(n=min(n_samples, len(df)), random_state=seed)

    num_t = torch.from_numpy(sample[nums].to_numpy(dtype=np.float32))
    cat_t = torch.from_numpy(sample[cats].to_numpy(dtype=np.int64))
    num_b = torch.zeros_like(num_t)
    cat_b = torch.zeros_like(cat_t)

    # ── Numerical: IntegratedGradients ──
    def fwd_num(x, cat_):
        """Forward fn for numerical attributions. cat_ is passed via additional_forward_args."""
        return model(x, cat_)

    ig = IntegratedGradients(fwd_num)
    attrs_num = ig.attribute(
        inputs=num_t,
        baselines=num_b,
        additional_forward_args=(cat_t,),
        n_steps=50,
        internal_batch_size=256
    )

    # ── Categorical: LayerIntegratedGradients ──
    cat_imp = []

    def make_fwd_cat(num_tensor, cat_tensor, idx):
        """Factory: returns a forward fn that swaps cat_tensor[:, idx] with x."""
        def fwd_cat(x):
            cm = cat_tensor.clone()
            cm[:, idx] = x.squeeze(1).long()
            return model(num_tensor, cm)
        return fwd_cat

    for i, emb in enumerate(model.embedding_d):
        fwd_cat = make_fwd_cat(num_t, cat_t, i)   # ← i bound at call time, not via closure
        lig = LayerIntegratedGradients(fwd_cat, layer=emb)
        attr_i = lig.attribute(
            inputs=cat_t[:, i].unsqueeze(1),
            baselines=cat_b[:, i].unsqueeze(1),
            n_steps=50,
            internal_batch_size=256
        )
        cat_imp.append(attr_i.sum(dim=-1).abs().mean().item())

    imp = pd.Series(np.concatenate([attrs_num.abs().mean(dim=0).numpy(), cat_imp]),
                    index=nums + cats)

    # --- TensorBoard: scalars + barplot ---
    for feat, val in imp.items():
        writer.add_scalar(f'captum/{feat}', val, 0)

    top5_conc = imp.sort_values(ascending=False).head(5).sum() / imp.abs().sum()
    writer.add_scalar('captum/top5_concentration', top5_conc, 0)
    writer.add_scalar('captum/total_attribution', imp.abs().sum(), 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    imp.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Mean |Attribution|')
    writer.add_figure('captum/barplot', fig, 0)
    plt.close(fig)

    writer.close()

    # --- Save for later cross-fold comparison ---
    Path('results').mkdir(exist_ok=True)
    imp.to_csv(f'results/fold{fold_idx}_importance.csv')
    model.to(DEVICE)
    return imp

#%% Set up data
def plot_data(train_d, valid_d, title="Training Progress"):
    """Plot training and validation metrics over epochs."""
    xx = np.arange(len(train_d))
    plt.figure(figsize=(10, 5))
    plt.plot(xx, train_d, label='Train', color='navy', marker='o', markersize=4)
    plt.plot(xx, valid_d, label='Validation', color='darkgreen', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return


# Split data
df_y = df_train[target].copy()
df_X = df_train.drop(columns=target)
Xtest = df_test.copy()

# Make meta-data
CATS = cats 
NUM = nums #+ nums_te
meta_data = {}
meta_data['NUM'] = NUM
meta_data['CATS'] = CATS
meta_data['num_cats'] = len(CATS)
meta_data['num_nums'] = len(NUM)
# Use category for embedding
# Made sure there are no "new" features in Xtest
meta_data = get_postsplit_meta(df_X, meta_data)



DEVICE = torch.device('cuda') # 'cpu') # 

#%%
kf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_state)

start_time = time.time()
oof = np.zeros(len(df_X))
preds = np.zeros((len(df_test),1))
fold_metric = []
all_importance = []
for i, (train_idx, valid_idx) in enumerate(kf.split(df_X, df_y)):
    print(f'#### FOLD {i+1} ####')
    writer = SummaryWriter(f"runs/nn_base/fold{i}")  # Start experiment TensorBoard writer
    # One line to log the entire experiment config to TensorBoard
    writer.add_text('config', json.dumps(asdict(cfg), indent=2), 0)

    Xtrain = df_X.loc[train_idx].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx].copy()
    yvalid = df_y.loc[valid_idx].copy()
         
    m = Xtrain[nums].mean()
    s = Xtrain[nums].std()
    Xtrain[nums] = (Xtrain[nums] - m) / s
    Xvalid[nums] = (Xvalid[nums] - m) / s
    Xtest[nums]  = (df_test[nums]  - m) / s
 
    # SET UP DATA standard dataset, dataloader functions
    traindataset = FastDataset(Xtrain, ytrain, meta_data['NUM'], meta_data['CATS'])
    validdataset = FastDataset(Xvalid, yvalid, meta_data['NUM'], meta_data['CATS'])
    trainloader = FastDataLoader(traindataset, batch_size=cfg.batch_size, shuffle=True)
    validloader = FastDataLoader(validdataset, batch_size=cfg.batch_size, shuffle=True)


    # DEF MODEL
    model = Model(meta_data, cfg.emb_dropout, cfg.hidden_dims, cfg.dropout_rates).to(DEVICE)
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    criterion = nn.BCEWithLogitsLoss().to(DEVICE) #nn.BCELoss() #   # neg wegiht / pos weight 0.2/0.8 # 
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    early_stopping = EarlyStopping(patience=cfg.patience)


    train_epoch_list = []
    valid_epoch_list = []
    for epoch in range(cfg.epochs):     
        train_loss = train(model, trainloader, optimizer, criterion, DEVICE)
        validation_loss, validation_metric, _ = valid(model, validloader, criterion, DEVICE) #, oof, val_idx)
        
        if epoch % 1 == 0:
            print(f'Epoch: {epoch}/{cfg.epochs}, Train loss: {train_loss:.6f}, Validation loss: {validation_loss:.6f}, Validation roc_auc: {validation_metric[0]:.6f}')

        train_epoch_list.append(train_loss)
        valid_epoch_list.append(validation_loss)

        # inside your epoch loop, replace plot_data with:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', validation_loss, epoch)
        writer.add_scalar('ROC-AUC/valid', validation_metric[0], epoch)
        writer.add_scalar('PR-AUC/valid', validation_metric[1], epoch)
        writer.add_scalar('F1/valid', validation_metric[2], epoch)
        # optionally log weight histograms
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    early_stopping.load_best_model(model)
    writer.close()  
    scheduler.step()
    
    # ── Captum + logging: capture the return value ──
    imp = log_fold_attributions(i, model, df_X.iloc[valid_idx],
                                nums, cats, m, s, DEVICE=DEVICE)
    all_importance.append(imp)
     
    #plot_data(train_epoch_list, valid_epoch_list)
    validation_loss, validation_metric, oof_pred = valid(model, validloader, criterion, DEVICE)
    print(f'Metric: {validation_metric[0]}')
    fold_metric.append(validation_metric[0])
    oof[valid_idx] = oof_pred.flatten()
    
    ydummy = pd.DataFrame(data=np.zeros((Xtest.shape[0],1)), columns=target) 
    testdataset = FastDataset(Xtest, ydummy, meta_data['NUM'], meta_data['CATS'])
    testloader = FastDataLoader(testdataset, batch_size=cfg.batch_size)
    y_pred = test_predictions(model, testloader, DEVICE)

    preds += y_pred/cfg.k_folds
    
    # Plot training progress for this fold
    plot_data(
        train_epoch_list, valid_epoch_list,
        title=f'NN Training Progress - Fold {i+1}'
    )
   
  
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(fold_metric)
print(f'Average metric: {np.mean(fold_metric)}')

#%%

# ── Cross-fold comparison ──
df_imp = pd.concat(all_importance, axis=1, keys=[f'fold{i}' for i in range(5)])
df_imp['mean'] = df_imp.mean(axis=1)
df_imp['std'] = df_imp.std(axis=1)
df_imp['cv'] = df_imp['std'] / df_imp['mean']

print(df_imp.sort_values('mean', ascending=False)[['mean', 'std', 'cv']]
      .to_string(float_format="%.4f"))


#%%


fname = 'nn2_base_test'

np.save(f'saved/{fname}_oof.npy', oof.reshape(-1,1))
np.save(f'saved/{fname}_preds.npy', preds)

#%%

print(oof.shape)
print(preds.shape)


#%%

df_sub = pd.read_csv('data/raw/sample_submission.csv')
df_sub['diagnosed_diabetes'] = preds
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)


# %%
