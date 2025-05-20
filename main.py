import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
import os

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Output directory
os.makedirs("outputs", exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Fake dataset for testing
train_dataset = torchvision.datasets.FakeData(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Load model
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Grad-CAM setup
grad_cam = LayerGradCam(model, model.layer4[-1])

# Training loop with OpenTelemetry
for epoch in range(1):
    with tracer.start_as_current_span(f"epoch_{epoch}") as epoch_span:
        for batch_idx, (images, labels) in enumerate(train_loader):
            with tracer.start_as_current_span(f"batch_{batch_idx}") as batch_span:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_span.set_attribute("loss", float(loss.item()))

                if batch_idx % 10 == 0:
                    input_img = images[0].unsqueeze(0)
                    input_img.requires_grad = True
                    attr = grad_cam.attribute(input_img, target=int(labels[0]))
                    attr = attr.squeeze().cpu().detach().numpy()

                    plt.imshow(attr, cmap='viridis')
                    plt.title(f"Epoch {epoch}, Batch {batch_idx} Grad-CAM")
                    plt.axis('off')
                    plt.savefig(f"outputs/gradcam_epoch{epoch}_batch{batch_idx}.png")
                    plt.close()

print("Training complete. Check ./outputs for Grad-CAM visualizations.")

