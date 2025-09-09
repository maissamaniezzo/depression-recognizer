# reprogram_train.py
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T, datasets
from torchvision.models import resnet18
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ===================== CONFIG =====================
REPO_ID     = "sebastiansarasti/MRIResnetModified"
FNAME       = "model.safetensors"
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3
EPS         = 4.0/255.0          # bound L_inf da perturbação δ
VAL_SPLIT   = 0.2
DATA_DIR    = "dados"            # deve conter 'controle/' e 'depressao/'
SEED        = 42
# ==================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED) # seed para ações randômicas pra manter o mais determinístico possível

# -------- 1) Carregar pesos do HF e montar a arquitetura (1 canal, 4 classes) --------
weights_path = hf_hub_download(repo_id=REPO_ID, filename=FNAME)

class MRIResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = resnet18(weights=None)
        # entrada 1 canal (grayscale)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # saída 4 classes (do treino de tumores)
        self.base_model.fc    = nn.Linear(512, 4)
    def forward(self, x): return self.base_model(x)

base = MRIResnet()
sd = load_file(weights_path)  # state_dict: dict nome->tensor

# missing (missing_keys): nomes de parâmetros que o seu modelo tem, mas não apareceram no arquivo de pesos.
# unexpected (unexpected_keys): nomes de pesos que existem no arquivo, mas o seu modelo não tem.
# should be: missing: [] unexpected: []
missing, unexpected = base.load_state_dict(sd, strict=False)
# print("missing:", missing, "unexpected:", unexpected)

# Congela o backbone inteiro (não recebem gradiente no backward)
for p in base.parameters():
    p.requires_grad = False
base.eval().to(device) # coloca modelo no modo avaliação e move modelo para cuda/cpu

# -------- 2) Data: ImageFolder (controle=0, depressao=1) + pré-processamento 1 canal --------
tfm = T.Compose([
    T.Grayscale(num_output_channels=1),  # força 1 canal (escala de cinza)
    T.Resize((IMG_SIZE, IMG_SIZE)),      # redimensiona p/ 224x224 (ou o que vc definiu)
    T.ToTensor(),                        # vira tensor [C,H,W] em float32 na faixa [0,1]
    T.Normalize(mean=[0.5], std=[0.5])   # normaliza cada canal: (x-0.5)/0.5  → ~[-1,1]
])

# pega imagem 2D, garante 1 canal e dimensões e transforma em tensor
full = datasets.ImageFolder(DATA_DIR, transform=tfm)

# quantas amostras vão para validação (p.ex. 20%)
n_val = int(len(full) * VAL_SPLIT)
# o restante vai para treino
n_tr  = len(full) - n_val

# separação dataset - quais exemplos pertencem a cada parte.
train_ds, val_ds = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(SEED))
# separação dataset - como esses exemplos são entregues ao modelo (em batches, embaralhados ou não).
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -------- 3) Componentes de reprogramação: δ e g(·) --------
# δ: perturbação universal (1 canal)
delta = nn.Parameter(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device))
# cabeçalho treinável (“MLP” mínima) 
# converte os 4 logits da sua ResNet (treinada em tumores) em 1 logit de “risco de depressão”.
g = nn.Linear(4, 1).to(device)

# método de otimização (atualiza parâmetros por gradiente)
opt = torch.optim.Adam(list(g.parameters()) + [delta], lr=LR)
# função de perda
bce = nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate(loader):
    base.eval(); g.eval()
    total_loss, total_n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.float().unsqueeze(1).to(device)  # [B] -> [B,1]
        d = torch.clamp(delta, -EPS, EPS)
        xrp = xb + d
        logits_dep = g(base(xrp))                # ok usar no_grad na avaliação
        loss = bce(logits_dep, yb)
        pred = (torch.sigmoid(logits_dep) >= 0.5).float()
        correct += (pred == yb).sum().item()
        total_loss += loss.item() * xb.size(0)
        total_n    += xb.size(0)
    return total_loss/total_n, correct/total_n

def train_epoch(loader):
    base.eval()    # backbone congelado
    g.train()      # cabeça treinável
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.float().unsqueeze(1).to(device)

        d = torch.clamp(delta, -EPS, EPS)
        xrp = xb + d

        # >>> sem no_grad aqui! precisamos do gradiente p/ δ
        logits_base = base(xrp)                  # [B,4]
        logits_dep  = g(logits_base)             # [B,1]
        loss = bce(logits_dep, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # projeta δ de volta ao intervalo após o passo
        with torch.no_grad():
            delta.clamp_(-EPS, EPS)

        total_loss += loss.item() * xb.size(0)
        total_n    += xb.size(0)

    return total_loss/total_n

best_val = 1e9
ckpt_path = "reprogram_ckpt.pth" # checkpoint do modelo

for epoch in range(1, EPOCHS+1):
    tr_loss = train_epoch(train_dl)
    val_loss, val_acc = evaluate(val_dl)
    print(f"[{epoch:02d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "delta": delta.detach().cpu(),
            "g_state": g.state_dict(),
            "repo_id": REPO_ID,
            "weights_file": FNAME,
            "img_size": IMG_SIZE,
            "eps": EPS,
            "normalize": {"mean":[0.5], "std":[0.5]},
        }, ckpt_path)
        print(f"  ✓ checkpoint salvo: {ckpt_path}")

# -------- 4) Inferência (probabilidade de depressão) --------
@torch.no_grad()
def predict_proba(pil_image, ckpt=ckpt_path):
    # mesmo pré-processamento do treino
    x = tfm(pil_image).unsqueeze(0).to(device)    # [1,1,H,W]
    ck = torch.load(ckpt, map_location=device)
    g.load_state_dict(ck["g_state"])
    d = torch.clamp(ck["delta"].to(device), -ck["eps"], ck["eps"])
    logits_dep = g(base(x + d))
    return torch.sigmoid(logits_dep).cpu().item()

# Exemplo:
# from PIL import Image
# img = Image.open("dados/depressao/qualquer.png")
# print("Prob. risco depressão:", predict_proba(img))
