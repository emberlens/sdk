# Train your own PyTorch model with a Nhan dataset.
# Needs PyTorch:  ./nhan install train
# Run:            ./nhan run examples/02_my_own_training.py path/to/your-export.zip
import sys
from pathlib import Path

import torch
import torchvision

import nhan
from nhan.torch_data import detection_collate

# No path given: use the sample export that ships with the SDK.
sample = Path(__file__).resolve().parent.parent / "fixtures" / "coco_joined"
ds = nhan.load(sys.argv[1] if len(sys.argv) > 1 else sample)
ds = ds.relabel("class")             # learn only the main class (Mango, Strawberry…)
train = ds.torch_detection("train")  # images + boxes in torchvision style
print("Classes:", train.classes)

loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, collate_fn=detection_collate)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=None, weights_backbone=None, num_classes=len(train.classes)
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model.train()
for epoch in range(2):
    for images, targets in loader:
        loss = sum(model(list(images), list(targets)).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1}: loss {loss.item():.3f}")
