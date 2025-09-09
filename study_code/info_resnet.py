import torch, torch.nn as nn
from torchvision.models import resnet18
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# 1) baixa os pesos do Hub
repo_id = "sebastiansarasti/MRIResnetModified"
weights = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

sd = load_file(weights)
print(sd)

# 2) camadas resnet
m = resnet18(weights=None)
m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 canal
m.fc    = nn.Linear(512, 4)  # 4 classes
print(m)  # imprime a pilha de blocos