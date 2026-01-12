# 1_ijepa_quick_test.py

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")

model_id = "facebook/ijepa_vith14_1k"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device).eval()

img = Image.open("image1.jpg").convert("RGB")
inputs = processor(img, return_tensors="pt")
inputs = {k: v.to(device) for k,v in inputs.items()}

with torch.no_grad():
    out = model(**inputs)

emb = out.last_hidden_state.mean(dim=1)
print("embedding shape:", tuple(emb.shape))
