from flask import Flask, render_template, request, jsonify
import torch
from torch.nn.functional import softmax
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from io import BytesIO
import warnings
import gdown
import os

# Suppress non-essential warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Function to download model from Google Drive
def download_model(model_url, output_path):
    if not os.path.exists(output_path):
        print("Downloading model from Google Drive...")
        gdown.download(model_url, output_path, quiet=False)
    else:
        print("Model file already exists.")

# Load model weights
model_path = 'model/model.pth'  # Local path to save model

# Correct Google Drive URL format
gdrive_url = 'https://drive.google.com/uc?id=140FTizWD_5ObuD_56_dHvTBbb6XX6-L_'

# Attempt to download the model
download_model(gdrive_url, model_path)

# Initialize ViT Image Processor and Model
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224', 
    num_labels=3, 
    ignore_mismatched_sizes=True
)

# Load model weights
try:
    with open(model_path, "rb") as model_file:
        buffer = BytesIO(model_file.read())
        # Load model with `weights_only=True` to avoid security warning
        state_dict = torch.load(buffer, map_location=torch.device('cpu'), weights_only=True)
        
    # Remove mismatched classifier layer weights
    keys_to_remove = ['classifier.weight', 'classifier.bias']
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# Define class labels
classes = ["Healthy", "septoria", "stripe_rust"]

# Prediction function with updated image preprocessing
def predict_image(image):
    # Use image processor to resize and preprocess the image
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1).squeeze()  # Apply softmax for probabilities
        predicted_class_idx = torch.argmax(probs).item()
        confidence = probs[predicted_class_idx].item() * 100
        
        all_probs = {classes[i]: round(probs[i].item() * 100, 2) for i in range(len(classes))}
        return classes[predicted_class_idx], confidence, all_probs

# Home page route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    try:
        # Open, convert, and preprocess image with image processor
        image = Image.open(BytesIO(file.read())).convert("RGB")
        predicted_class, confidence, all_probs = predict_image(image)
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence_percentage": round(confidence, 2),
            "class_probabilities": all_probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))

