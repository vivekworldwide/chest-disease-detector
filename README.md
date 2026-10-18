# 🫁 ChestVision AI — Chest X-Ray Disease Classifier

**Real-time chest radiograph analysis** powered by a custom **Hybrid CNN + Vision Transformer** architecture built with PyTorch.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 🔬 What It Does

Upload a chest X-ray (PA/AP view) and get:
- **Disease Classification**: COVID-19 · Normal · Viral Pneumonia
- **Confidence Score**: Model certainty percentage
- **Probability Distribution**: Breakdown across all classes
- **Clinical Report**: AI-generated findings & recommendations
- **Input Validation**: Rejects non-X-ray images (photos, documents, screenshots)

## 🏗️ Architecture

| Component | Detail |
|---|---|
| **Backbone** | 5-Stage CNN (3→32→64→128→256→512 channels) |
| **Attention** | 2-Layer Transformer Encoder (8-head self-attention) |
| **Input** | 224×224 RGB tensor, ImageNet-normalized |
| **Output** | 3-class Softmax probability distribution |
| **Framework** | PyTorch · FastAPI · Uvicorn |

## 📊 Training

- **Dataset**: COVID-19 Radiography Database (3 classes)
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=3)
- **Augmentation**: Random rotation, horizontal flip, color jitter, affine transforms

## 🛡️ Safety

- Out-of-domain image detection (rejects non-X-ray uploads)
- Medical disclaimer included — not a substitute for professional diagnosis

## 🚀 Local Development

```bash
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860
```

## ☁️ Deployment

Deployed on **Render** (free tier). Connect your GitHub repo and it auto-deploys on every push.

---

> ⚠️ **Disclaimer**: This tool is for educational and research purposes only. It is NOT a certified medical device. Always consult qualified healthcare professionals for medical diagnosis.
