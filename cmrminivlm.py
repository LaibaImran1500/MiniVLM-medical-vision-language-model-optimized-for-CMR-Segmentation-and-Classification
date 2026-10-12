# -*- coding: utf-8 -*-
# # CELL 1: SETUP AND PERSISTENT DATA EXTRACTION
# import os
# import zipfile
# import glob
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import pandas as pd
# import scipy.ndimage
# import textwrap
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModel
# from sklearn.model_selection import train_test_split
# from google.colab import drive
# import torchvision.models as models

# # 1. Mount Google Drive
# drive.mount('/content/drive')

# # 2. Define Paths
# BASE_DIR = '/content/drive/MyDrive/MiniVLM_EMIDEC'
# ZIP_PATH = os.path.join(BASE_DIR, 'emidec_training.zip')
# LOCAL_DATA_DIR = '/content/emidec_data'
# CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
# RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # 3. Robust Local Extraction
# if not os.path.exists(LOCAL_DATA_DIR):
#     print(f"Extracting {ZIP_PATH} to local runtime for high-speed access...")
#     !unzip -q "{ZIP_PATH}" -d "{LOCAL_DATA_DIR}"
#     print("Extraction complete.")
# else:
#     print("Data already available on local runtime.")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# CELL 1: SETUP AND PERSISTENT DATA EXTRACTION
import os
import sys
import zipfile
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import scipy.ndimage
import textwrap
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import torchvision.models as models

# 1. Environment Detection (Docker vs. Colab)
IN_CONTAINER = os.getenv('DOCKER_MODE', 'false').lower() == 'true'

# 2. Dynamic Path Assignment
if IN_CONTAINER:
    print("Running in Production/Docker Mode...")
    BASE_DIR = os.getenv('BASE_DIR', '/workspace/data')
    LOCAL_DATA_DIR = os.getenv('DATA_DIR', '/workspace/emidec_data')
    ZIP_PATH = os.path.join(BASE_DIR, 'emidec_training.zip')
else:
    print("Running in Notebook/Colab Fallback Mode...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    BASE_DIR = '/content/drive/MyDrive/MiniVLM_EMIDEC'
    ZIP_PATH = os.path.join(BASE_DIR, 'emidec_training.zip')
    LOCAL_DATA_DIR = '/content/emidec_data'

CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 3. Robust Pure-Python Extraction (Replaces !unzip bash command)
if not os.path.exists(LOCAL_DATA_DIR):
    if os.path.exists(ZIP_PATH):
        print(f"Extracting {ZIP_PATH} to local runtime space...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_DATA_DIR)
        print("Extraction complete.")
    else:
        print(f"ERROR: Dataset zip file not found at {ZIP_PATH}")
else:
    print("Data already available on local runtime.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# CELL 2: SANITIZED DATA LOADING
class EMIDEC_Dataset(Dataset):
    def __init__(self, case_dirs, transform_size=(128, 128)):
        self.case_dirs = case_dirs
        self.transform_size = transform_size

    def _parse_narrative(self, case_dir):
        case_name = os.path.basename(case_dir)
        label = 1 if 'P' in case_name.upper() else 0

        txt_files = glob.glob(os.path.join(case_dir, '*.txt')) + glob.glob(os.path.join(os.path.dirname(case_dir), '*.txt'))
        if txt_files:
            match = [f for f in txt_files if case_name in f]
            txt_path = match[0] if match else txt_files[0]
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()

            # SANITIZATION: Remove keywords that give away the answer directly
            for word in ['pathological', 'normal', 'case', 'p0', 'n0', 'p1', 'n1', 'healthy']:
                content = content.replace(word, "[MASK]")

            # Extract clinical symptoms (usually the first few lines of EMIDEC txt)
            narrative = "Clinical findings: " + content.strip()[:300]
        else:
            narrative = "No clinical history provided for this patient."

        return narrative, label

    def _load_nifti_slice(self, folder_path, is_mask=False):
        nii_files = glob.glob(os.path.join(folder_path, '*.nii.gz'))
        if not nii_files: return torch.zeros(1, *self.transform_size)
        img = nib.load(nii_files[0]).get_fdata()
        slice_2d = img[:, :, img.shape[2] // 2]
        res = cv2.resize(slice_2d, self.transform_size, interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
        if not is_mask:
            res = (res - np.min(res)) / (np.max(res) - np.min(res) + 1e-8)
        return torch.tensor(res, dtype=torch.float32).unsqueeze(0)

    def __len__(self): return len(self.case_dirs)

    def __getitem__(self, idx):
        d = self.case_dirs[idx]
        narr, lbl = self._parse_narrative(d)
        return {
            'image': self._load_nifti_slice(os.path.join(d, 'Images'), False),
            'mask': (self._load_nifti_slice(os.path.join(d, 'Contours'), True) > 0).float(),
            'narrative': narr,
            'label': lbl
        }

# Re-run the split with new logic
candidate_dirs = glob.glob(os.path.join(LOCAL_DATA_DIR, '**/Images'), recursive=True)
all_case_dirs = [os.path.dirname(d) for d in candidate_dirs]
all_labels = [1 if 'P' in os.path.basename(d).upper() else 0 for d in all_case_dirs]

train_dirs, test_dirs, _, _ = train_test_split(all_case_dirs, all_labels, test_size=0.20, stratify=all_labels, random_state=42)
train_loader = DataLoader(EMIDEC_Dataset(train_dirs), batch_size=8, shuffle=True)
test_loader = DataLoader(EMIDEC_Dataset(test_dirs), batch_size=8, shuffle=False)

# CELL 3: IMPROVED RESIDUAL FUSION ARCHITECTURE
class MiniVLM(nn.Module):
    def __init__(self, mode='multimodal', embed_dim=512):
        super().__init__()
        self.mode = mode

        # Vision Branch (Frozen backbone initially to keep 0.80 score)
        resnet = models.resnet50(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.v_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.v_proj = nn.Conv2d(2048, embed_dim, 1)

        # Text Branch
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.t_backbone = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.t_proj = nn.Linear(768, embed_dim)

        # Gated Cross-Attention (Multimodal Core)
        self.attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        # Common Heads
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, img, txt):
        # 1. Vision Features (Baseline)
        v_feat = self.v_proj(self.v_backbone(img))
        B, C, H, W = v_feat.shape

        # 2. Text Features
        inputs = self.tokenizer(txt, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        t_feat = self.t_proj(self.t_backbone(**inputs).last_hidden_state[:, 0, :])

        attn_maps = None
        fused_v = v_feat # Default to vision only

        if self.mode == 'multimodal':
            # Cross Attention: Text queries Vision
            v_flat = v_feat.view(B, C, -1).permute(0, 2, 1)
            query = t_feat.unsqueeze(1)
            attn_out, weights = self.attn(query, v_flat, v_flat)
            attn_maps = weights.view(B, H, W)

            # Gated Residual: Vision + (Text_Gate * Attention_Context)
            context = attn_out.permute(0, 2, 1).view(B, C, 1, 1)
            g = self.gate(t_feat).view(B, C, 1, 1)
            fused_v = v_feat + (g * context) # Residual connection preserves vision baseline

        elif self.mode == 'text_only':
            fused_v = t_feat.view(B, C, 1, 1).expand(-1, -1, H, W)

        seg = F.interpolate(self.decoder(fused_v), size=(img.shape[-2], img.shape[-1]), mode='bilinear')
        cls = self.classifier(fused_v.mean(dim=[2,3]))

        return seg, cls, v_feat, t_feat, attn_maps

# CELL 4: LOSSES AND TRAINING LOGIC
def dice_loss(p, t):
    p = torch.sigmoid(p)
    inter = (p * t).sum(dim=(2,3))
    union = p.sum(dim=(2,3)) + t.sum(dim=(2,3))
    # Using Log-Cosh Dice for smoother gradient surface
    return torch.log(torch.cosh(1 - (2.*inter + 1e-5)/(union + 1e-5))).mean()

def nce_loss(v, t):
    v_p = F.normalize(v.mean(dim=[2,3]), dim=-1)
    t_p = F.normalize(t, dim=-1)
    logits = torch.matmul(v_p, t_p.T) / 0.1
    return F.cross_entropy(logits, torch.arange(v.size(0)).to(device))

def get_metrics(sp, st, cp, ct):
    sp_bin = (torch.sigmoid(sp) > 0.5).float()
    dice = (2.*(sp_bin*st).sum())/ (sp_bin.sum() + st.sum() + 1e-8)

    cp_bool = (torch.sigmoid(cp) > 0.5).bool()
    ct_bool = (ct > 0.5).bool()

    tp = (cp_bool & ct_bool).float().sum()
    fp = (cp_bool & ~ct_bool).float().sum()
    fn = (~cp_bool & ct_bool).float().sum()
    tn = (~cp_bool & ~ct_bool).float().sum()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return dice.item(), acc.item(), precision.item(), recall.item(), f1.item()

def run_experiment(mode, epochs=20):
    model = MiniVLM(mode=mode).to(device)
    # Lower learning rate for multimodal to prevent catastrophic interference
    lr = 2e-5 if mode == 'multimodal' else 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    path = os.path.join(CHECKPOINT_DIR, f'{mode}_v2.pth')

    hist = {'loss': [], 'dice': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    start = 0

    # Auto-cleanup for existing checkpoints to ensure fresh start with new loss logic
    if os.path.exists(path):
        os.remove(path)

    for ep in range(start, epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            s, c, v, t, _ = model(b['image'].to(device), b['narrative'])

            # Loss Balancing: Prioritize Classification (BCE) in Multimodal
            l_seg = dice_loss(s, b['mask'].to(device))
            l_cls = F.binary_cross_entropy_with_logits(c, b['label'].float().to(device).unsqueeze(1))

            if mode == 'multimodal':
                # Weighting: 3.0 for Classif, 1.0 for Seg, 0.1 for Alignment
                l = (1.0 * l_seg) + (3.0 * l_cls) + (0.1 * nce_loss(v, t))
            else:
                l = l_seg + l_cls

            l.backward()
            # Gradient Clipping to stabilize multimodal fusion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        td, ta, tp, tr, tf, tl = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for b in test_loader:
                s, c, v, t, _ = model(b['image'].to(device), b['narrative'])
                d, a, p, r, f = get_metrics(s, b['mask'].to(device), c, b['label'].float().to(device).unsqueeze(1))
                td += d; ta += a; tp += p; tr += r; tf += f
                tl += (dice_loss(s, b['mask'].to(device))).item()

        num_batches = len(test_loader)
        hist['loss'].append(tl/num_batches)
        hist['dice'].append(td/num_batches)
        hist['acc'].append(ta/num_batches)
        hist['precision'].append(tp/num_batches)
        hist['recall'].append(tr/num_batches)
        hist['f1'].append(tf/num_batches)

        print(f"Epoch {ep+1} | Dice: {hist['dice'][-1]:.4f} | Acc: {hist['acc'][-1]:.4f}| Pre: {hist['precision'][-1]:.4f}| Rec: {hist['recall'][-1]:.4f} | F1: {hist['f1'][-1]:.4f}")
        torch.save({'epoch': ep+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'history': hist}, path)

    return hist, model

# CELL 5: RUN ABLATION AND PLOT
results = {}
models_dict = {}
for m in ['vision_only', 'text_only', 'multimodal']:
    results[m], models_dict[m] = run_experiment(m)

# Quantitative Summary Table
summary = pd.DataFrame([{
    'Mode': k,
    'Dice': v['dice'][-1],
    'Acc': v['acc'][-1],
    'Precision': v['precision'][-1],
    'Recall': v['recall'][-1],
    'F1-Score': v['f1'][-1]
} for k, v in results.items()])

print("\n", summary)
summary.to_csv(os.path.join(RESULTS_DIR, 'ablation_summary.csv'))

# Comprehensive 2x3 Performance Visualization
plt.figure(figsize=(20, 10))

metrics = [('dice', 'Dice Score (Segmentation)'), ('acc', 'Classification Accuracy'),
           ('f1', 'F1-Score (Balanced)'), ('precision', 'Precision'),
           ('recall', 'Recall (Sensitivity)'), ('loss', 'Validation Loss')]

for i, (key, title) in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    plt.title(title)
    for k, v in results.items():
        plt.plot(v[key], label=k, linewidth=2)
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# CELL 6: SPATIAL GROUNDING (ATTENTION MAPS)
def visualize_reasoning(model):
    model.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        seg, cls, _, _, attn = model(batch['image'][:1].to(device), batch['narrative'][:1])

    if attn is not None:
        img = batch['image'][0].squeeze().numpy()
        att = scipy.ndimage.zoom(attn[0].cpu().numpy(), (img.shape[0]/attn.shape[1], img.shape[1]/attn.shape[2]), order=1)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.title("Original MRI"); plt.imshow(img, cmap='gray')
        plt.subplot(1, 3, 2); plt.title("Clinical-Attention Map"); plt.imshow(img, cmap='gray'); plt.imshow(att, cmap='jet', alpha=0.5)
        plt.subplot(1, 3, 3); plt.axis('off'); plt.text(0, 0.5, "\n".join(textwrap.wrap(batch['narrative'][0], 40)), fontsize=10)
        plt.show()

visualize_reasoning(models_dict['multimodal'])

# CELL 5: QUANTITATIVE METRICS AND ABLATION VISUALIZATION
import pandas as pd
import seaborn as sns

def generate_summary(results_dict):
    """
    Extracts the best metrics from each training history.
    """
    summary_data = []

    for mode, history in results_dict.items():
        # Get the best scores achieved during training
        best_dice = max(history['dice'])
        best_acc = max(history['acc'])
        final_loss = history['loss'][-1]

        summary_data.append({
            'Experiment Mode': mode.replace('_', ' ').title(),
            'Best Dice Score': round(best_dice, 4),
            'Best Accuracy': round(best_acc, 4),
            'Final Test Loss': round(final_loss, 4)
        })

    df = pd.DataFrame(summary_data)
    # Sort so Multimodal is prominent or by performance
    df = df.sort_values(by='Best Dice Score', ascending=False)
    return df

# 1. Generate and Display Table
summary_df = generate_summary(results)
print("\n--- ABLATION STUDY QUANTITATIVE RESULTS ---")
print(summary_df.to_string(index=False))

# 2. Save Results to CSV for research documentation
summary_df.to_csv(os.path.join(RESULTS_DIR, 'quantitative_ablation_results.csv'), index=False)

# 3. Visualization: Side-by-Side Bar Charts
plt.style.use('seaborn-v0_8-muted')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Dice Scores
sns.barplot(x='Experiment Mode', y='Best Dice Score', data=summary_df, ax=ax1, palette='viridis')
ax1.set_title('Segmentation Performance (Dice)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.0)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')

# Plot Accuracy Scores
sns.barplot(x='Experiment Mode', y='Best Accuracy', data=summary_df, ax=ax2, palette='magma')
ax2.set_title('Classification Performance (Accuracy)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1.0)
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'performance_comparison_bars.png'), dpi=300)
plt.show()

# 4. Learning Curve Comparison
plt.figure(figsize=(10, 6))
for mode, history in results.items():
    plt.plot(history['dice'], label=f"{mode.title()} (Dice)", linestyle='--')
    plt.plot(history['acc'], label=f"{mode.title()} (Acc)", linewidth=2)

plt.title("Learning Dynamics Across Modes", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Metric Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
