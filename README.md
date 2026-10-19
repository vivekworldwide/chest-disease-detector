# 🫁 ChestVision AI — Chest X-Ray Disease Classifier

**High-Performance Pulmonary Diagnostic Dashboard** powered by a custom **Hybrid CNN + Vision Transformer** deep learning model built with PyTorch & FastAPI.

[![Website](https://render.com/images/deploy-to-render-button.svg)](https://chestvision-ai-5a9b.onrender.com/)

---

## 🔬 Core Capabilities

- **Frontal Radiograph Screening**: Automated detection for **COVID-19**, **Normal (Healthy)**, and **Viral Pneumonia**.
- **Confidence Metrics & Distribution**: Instant percentage confidence and class probability distribution breakdown.
- **Clinical Findings & Recommendations**: Automated synthesis of clinical notes and next-step actions.
- **Domain Guard / Out-Of-Domain Validation**: Automatically detects and rejects non-X-ray images (such as color photos, text documents, or UI screenshots) preventing false-positive predictions.
- **Sub-100ms Inference**: Lightweight CPU-optimized PyTorch execution ideal for free-tier cloud deployment.

---

## 🏗️ Architecture

| Component | Specification |
|---|---|
| **Spatial Feature Backbone** | 5-Stage Convolutional Neural Network (3 → 32 → 64 → 128 → 256 → 512 channels) |
| **Global Context Modeling** | 2-Layer Transformer Encoder with 8-Head Multi-Head Self-Attention & Learnable Positional Embeddings |
| **Input Resolution** | 224 × 224 RGB, ImageNet standard normalization |
| **Classification Head** | Dense projection with ReLU and Dropout (0.2) into 3 Softmax output logits |
| **Serving Stack** | FastAPI + Uvicorn (Asynchronous ASGI server) |

---

## 💻 Local Execution

To run locally on your machine:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch FastAPI app
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` (or `http://localhost:5000`).
Interactive Swagger API documentation is available at `http://127.0.0.1:5000/docs`.

---

> ⚠️ **Clinical Disclaimer**: ChestVision AI is designed for research, academic demonstration, and educational purposes. It is not an FDA/CE-cleared diagnostic device. All assessments should be corroborated by clinical evaluation and certified medical professionals.
