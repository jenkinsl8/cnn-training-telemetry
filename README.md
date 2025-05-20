# CNN Training Telemetry with OpenTelemetry and Grad-CAM

This project provides a simple example of how to observe what a CNN model is learning in real-time during training using:

* **PyTorch** for model training
* **Captum** for Grad-CAM visualizations
* **OpenTelemetry** for lightweight tracing/logging

It generates Grad-CAM heatmaps and OpenTelemetry-style span logs for select training batches.

---

## 📦 Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the training script with:

```bash
python main.py
```

This will:

* Train a simple ResNet18 model on fake image data
* Log loss per batch using OpenTelemetry (console output)
* Generate Grad-CAM images every 10 batches and save them to `./outputs/`

You can replace the dataset and model with your own by modifying `main.py`.

---

## 🗂 Outputs

* **Console logs**: Span traces showing batch-level loss.
* **Grad-CAM images**: Visual explanations of what the model is focusing on (saved to `outputs/`).

Example image file:

```
outputs/gradcam_epoch0_batch0.png
```

---

## 🔄 Customization

* Swap `FakeData` with your own dataset (e.g., CIFAR10, ImageNet).
* Replace `resnet18` with your own model.
* Add more telemetry (e.g., learning rate changes, accuracy) using `batch_span.set_attribute()`.

---

## 🧠 Purpose

This setup mimics an OpenTelemetry-style monitoring system for deep learning, helping you visualize and understand model learning behavior as it trains.

Useful for:

* Debugging CNN performance
* Educational visualization
* Model observability research

---

## 🔗 License

MIT License

