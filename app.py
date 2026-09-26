import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image

from model import ChestDiseaseClassifier

# 1. Initialize FastAPI Application with metadata for automatic Swagger UI (/docs)
app = FastAPI(
    title="ChestVision AI API",
    description="High-performance asynchronous API for automated Chest X-ray disease classification (COVID, Normal, Viral Pneumonia) using PyTorch Hybrid CNN-Transformer.",
    version="1.0.0"
)

# 2. Mount Static Files & Template Engine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# 3. Load Trained Model Checkpoint
MODEL_PATH = os.path.join(BASE_DIR, "chest_disease_model.pth")
classifier = None

try:
    print(f"Loading Chest Disease AI Model from: {MODEL_PATH}")
    classifier = ChestDiseaseClassifier(MODEL_PATH)
    print(f"Model successfully loaded on device: {classifier.device}")
except Exception as e:
    print(f"Warning: Model could not be loaded on startup ({e}). It will initialize on request.")


class PredictJSONRequest(BaseModel):
    sample: Optional[str] = None
    image_base64: Optional[str] = None


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse, summary="Serve Web Interface")
async def index(request: Request):
    """
    Renders the primary diagnostic dashboard.
    """
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health", summary="API Health Check")
async def health():
    """
    Returns system health, model load state, target classes, and PyTorch device.
    """
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "classes": classifier.classes if classifier else [],
        "device": str(classifier.device) if classifier else "unknown"
    }


@app.get("/sample/{filename}", summary="Get Sample Radiograph")
async def get_sample(filename: str):
    """
    Serves verified chest X-ray sample files for immediate testing.
    """
    sample_path = os.path.join(BASE_DIR, "static", "samples", filename)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample file {filename} not found.")
    return FileResponse(sample_path)


@app.post("/predict", summary="Classify Chest X-Ray")
async def predict(
    request: Request,
    image: Optional[UploadFile] = File(None)
):
    """
    Analyzes an uploaded chest X-ray image (multipart file or JSON sample name)
    and returns predictions, confidence, probability distribution, and clinical recommendations.
    """
    global classifier
    if classifier is None:
        try:
            classifier = ChestDiseaseClassifier(MODEL_PATH)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Model failed to initialize: {str(e)}"}
            )

    image_bytes = None
    source_name = "Uploaded Scan"

    # Case 1: Uploaded file via multipart form-data
    if image is not None:
        if not is_allowed_file(image.filename):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Unsupported file format. Please upload a PNG, JPG, or JPEG image."}
            )
        source_name = image.filename
        image_bytes = await image.read()

    # Case 2: JSON payload (sample image name or base64)
    else:
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.json()
                sample_file = body.get("sample")
                base64_str = body.get("image_base64")

                if sample_file:
                    sample_path = os.path.join(BASE_DIR, "static", "samples", sample_file)
                    if not os.path.exists(sample_path):
                        return JSONResponse(
                            status_code=404,
                            content={"success": False, "error": f"Sample {sample_file} not found."}
                        )
                    source_name = sample_file
                    with open(sample_path, "rb") as f:
                        image_bytes = f.read()

                elif base64_str:
                    if "," in base64_str:
                        base64_str = base64_str.split(",", 1)[1]
                    image_bytes = base64.b64decode(base64_str)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid request payload: {str(e)}"}
            )

    if not image_bytes:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No valid chest X-ray image received."}
        )

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Create thumbnail preview
        preview_buffer = io.BytesIO()
        pil_img.save(preview_buffer, format="JPEG", quality=85)
        preview_base64 = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.getvalue()).decode("utf-8")

        # Execute PyTorch inference
        result = classifier.predict(pil_img)

        return {
            "success": True,
            "filename": source_name,
            "image_preview": preview_base64,
            "prediction": result["predicted_class"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
            "device": result["device"],
            "probabilities": result["probabilities"],
            "metadata": result["metadata"]
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error during image processing: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    # Run locally with Uvicorn server on port 5000
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
