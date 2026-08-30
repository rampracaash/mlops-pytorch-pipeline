import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from model import get_model
from dataset import get_transforms
import yaml
from pathlib import Path

app = FastAPI(title="PyTorch Model Serving")

# Global variables for model and config
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transforms_fn = get_transforms(train=False)

def load_config():
    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")
    
    if not config_path.exists():
        # Default config if file doesn't exist during serving
        return {
            "model": {"architecture": "resnet18", "num_classes": 10},
            "output": {"checkpoint_dir": "/app/checkpoints", "model_name": "classifier_v1.pt"}
        }
        
    with open(config_path) as f:
        return yaml.safe_load(f)

@app.on_event("startup")
async def startup_event():
    global model
    config = load_config()
    
    # Initialize model
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"]
    ).to(device)
    
    # Load weights
    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / config["output"]["model_name"]
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using uninitialized weights.")

@app.get("/health")
async def health_check():
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = transforms_fn(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Format results
        class_probs = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        predicted_class = str(torch.argmax(probabilities).item())
        
        return JSONResponse({
            "predicted_class": predicted_class,
            "probabilities": class_probs
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
