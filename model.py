import os
import io
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class CNNBackbone(nn.Module):
    """
    5-stage Convolutional Neural Network backbone for localized 
    spatial feature extraction from Chest X-ray images.
    """
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.features = nn.Sequential(
            # Stage 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)


class ChestDiseaseModel(nn.Module):
    """
    Hybrid CNN + Vision Transformer model.
    CNN extracts patch feature maps, which are flattened into visual tokens,
    augmented with learnable positional embeddings, and processed through
    Transformer Self-Attention layers for global pulmonary context modeling.
    """
    def __init__(self, num_classes=3, d_model=512, nhead=8, num_transformer_layers=2):
        super(ChestDiseaseModel, self).__init__()
        self.cnn = CNNBackbone()
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.cnn(x)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embedding
        trans_out = self.transformer(tokens)
        pooled = trans_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


DISEASE_METADATA = {
    "Normal": {
        "badge_class": "status-normal",
        "icon": "shield-check",
        "severity": "Clear / Low Risk",
        "summary": "No significant radiographic abnormalities, consolidations, or active infiltrates were detected in the pulmonary fields.",
        "findings": [
            "Clear lung parenchyma with bilateral symmetric aeration.",
            "Normal cardiothoracic ratio without signs of cardiomegaly.",
            "Costophrenic angles are sharp and unobstructed."
        ],
        "recommendations": [
            "Maintain routine health monitoring and standard preventative wellness habits.",
            "If respiratory symptoms (cough, shortness of breath) persist, consult a physician for clinical evaluation."
        ]
    },
    "COVID": {
        "badge_class": "status-covid",
        "icon": "virus",
        "severity": "High Attention Required",
        "summary": "Radiographic patterns characteristic of COVID-19 bronchopulmonary involvement observed, commonly marked by bilateral peripheral ground-glass opacities.",
        "findings": [
            "Bilateral patchy ground-glass opacities predominantly in lower and peripheral lung zones.",
            "Subpleural consolidations and vascular thickening consistent with viral pneumonitis.",
            "Possible mild interstitial markings or crazy-paving patterns."
        ],
        "recommendations": [
            "Confirm diagnosis immediately via RT-PCR or rapid antigen testing.",
            "Monitor oxygen saturation (SpO2) using a pulse oximeter regularly.",
            "Isolate to prevent viral transmission and seek urgent medical care if breathlessness develops."
        ]
    },
    "Viral Pneumonia": {
        "badge_class": "status-pneumonia",
        "icon": "lungs",
        "severity": "Moderate to High Attention",
        "summary": "Opacities and interstitial infiltrates suggestive of viral pneumonia detected across one or both pulmonary zones.",
        "findings": [
            "Diffuse peribronchial thickening and interstitial markings.",
            "Patchy reticular or bronchopneumonic opacities across pulmonary segments.",
            "Absence of dense lobar consolidation typical of pure bacterial pneumonia."
        ],
        "recommendations": [
            "Consult a pulmonologist or healthcare professional for clinical auscultation and diagnosis.",
            "Follow supportive respiratory care, hydration, and prescribed antivirals or bronchodilators if indicated.",
            "Seek urgent evaluation if chest pain, sustained high fever, or wheezing worsens."
        ]
    }
}


class ChestDiseaseClassifier:
    """
    Production inference engine that handles checkpoint loading,
    standardized image transforms, class prediction, and clinical metadata synthesis.
    """
    def __init__(self, checkpoint_path="chest_disease_model.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.classes = ["COVID", "Normal", "Viral Pneumonia"]
        self.model = ChestDiseaseModel(num_classes=len(self.classes))
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "classes" in checkpoint:
                self.classes = checkpoint["classes"]
        else:
            self.model.load_state_dict(checkpoint)

    def predict(self, image_input):
        """
        Runs inference on a PIL Image or bytes.
        Returns detailed prediction dictionary.
        """
        start_time = time.time()

        if isinstance(image_input, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            raise ValueError("Unsupported image input type.")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, dim=0)

        pred_class = self.classes[pred_idx.item()]
        conf_percent = round(confidence.item() * 100, 2)
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        # Build class-level probability breakdown
        prob_distribution = []
        for i, class_name in enumerate(self.classes):
            score = round(probabilities[i].item() * 100, 2)
            prob_distribution.append({
                "class_name": class_name,
                "percentage": score,
                "is_primary": (class_name == pred_class)
            })

        # Sort probability breakdown descending
        prob_distribution.sort(key=lambda item: item["percentage"], reverse=True)

        meta = DISEASE_METADATA.get(pred_class, {
            "badge_class": "status-unknown",
            "icon": "activity",
            "severity": "Clinical Review Needed",
            "summary": "AI prediction complete. Please review with medical team.",
            "findings": ["Analysis generated based on deep CNN-Transformer features."],
            "recommendations": ["Consult medical provider for official assessment."]
        })

        return {
            "predicted_class": pred_class,
            "confidence": conf_percent,
            "latency_ms": elapsed_ms,
            "device": str(self.device),
            "probabilities": prob_distribution,
            "metadata": meta
        }
