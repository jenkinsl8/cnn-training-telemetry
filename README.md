# cnn\_training\_telemetry/main.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
import numpy as np
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
import os

# Setup OpenTelemetry

trace.set\_tracer\_provider(TracerProvider())
tracer = trace.get\_tracer(**name**)
trace.get\_tracer\_provider().add\_span\_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Create output directory

os.makedirs("outputs", exist\_ok=True)

# Data transforms and loader

transform = transforms.Compose(\[
transforms.Resize((64, 64)),
transforms.ToTensor(),
])

train\_dataset = torchvision.datasets.FakeData(transform=transform, num\_classes=2)
train\_loader = DataLoader(train\_dataset, batch\_size=8, shuffle=True)

# Model setup

model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in\_features, 2)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is\_available() else "cpu")
model.to(device)

# Grad-CAM setup

grad\_cam = LayerGradCam(model, model.layer4\[-1])

# Training loop with telemetry and Grad-CAM visualization

for epoch in range(1):
with tracer.start\_as\_current\_span(f"epoch\_{epoch}") as epoch\_span:
for batch\_idx, (images, labels) in enumerate(train\_loader):
with tracer.start\_as\_current\_span(f"batch\_{batch\_idx}") as batch\_span:
images, labels = images.to(device), labels.to(device)
outputs = model(images)
loss = criterion(outputs, labels)

```
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_span.set_attribute("loss", float(loss.item()))

            if batch_idx % 10 == 0:
                input_img = images[0].unsqueeze(0)
                input_img.requires_grad = True
                attr = grad_cam.attribute(input_img, target=int(labels[0]))
                attr = attr.squeeze().cpu().detach().numpy()

                input_img_np = input_img.squeeze().cpu().permute(1, 2, 0).detach().numpy()
                input_img_np = (input_img_np - input_img_np.min()) / (input_img_np.max() - input_img_np.min())

                heatmap = (attr - attr.min()) / (attr.max() - attr.min())

                # Overlay heatmap on input image
                fig, ax = plt.subplots()
                ax.imshow(input_img_np)
                ax.imshow(heatmap, cmap='viridis', alpha=0.5)
                ax.set_title(f"Epoch {epoch}, Batch {batch_idx} Grad-CAM Overlay")
                ax.axis('off')
                plt.savefig(f"outputs/overlay_epoch{epoch}_batch{batch_idx}.png")
                plt.close()
```

print("Training complete. Overlay visualizations saved in ./outputs")

