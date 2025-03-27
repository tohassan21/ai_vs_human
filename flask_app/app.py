import boto3
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS credentials 
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MODEL_PATH = "models/model_05.h5"  
LOCAL_MODEL_PATH = "/tmp/model_05.h5"  

# Set LOCAL_MODEL_PATH based on OS
if os.name == 'nt':  # Windows
    # Create 'tmp' folder in your current working directory if not exists
    LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "tmp", "model_05.h5")
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
else:
    # On Heroku, use the absolute path in /tmp
    LOCAL_MODEL_PATH = "/tmp/model_05.h5"

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Download model if not present locally
if not os.path.exists(LOCAL_MODEL_PATH):
    print("Downloading model from S3...")
    s3.download_file(S3_BUCKET_NAME, MODEL_PATH, LOCAL_MODEL_PATH)

# Load the model
print("Loading model...")
model = load_model(LOCAL_MODEL_PATH)
print("Model loaded successfully.")

# Flask app
app = Flask(__name__)
CORS(app)

# 2 MB max file size
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Convert the uploaded file to a BytesIO object
        img_bytes = BytesIO(file.read())

        # Load and preprocess the image
        img = image.load_img(img_bytes, target_size=(255, 255))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255

        # Make prediction (get raw output from the model)
        preds = model.predict(img_array)
        
        # Debug: log raw prediction values
        print("Raw prediction:", preds)  

        # Check output shape to decide which branch to use
        if preds.shape[-1] == 1:
            # Model with a single output neuron
            raw_value = preds[0][0]
            # Apply sigmoid manually to convert to probability
            prob = tf.nn.sigmoid(raw_value).numpy()
            if prob >= 0.5:
                label = "Human-generated"
                confidence = prob * 100
            else:
                label = "AI-generated"
                confidence = (1 - prob) * 100

        elif preds.shape[-1] == 2:
            # Model with two outputs; force softmax normalization
            probabilities = tf.nn.softmax(preds[0]).numpy()
            label_index = np.argmax(probabilities)
            confidence = float(probabilities[label_index]) * 100
            label = "Human-generated" if label_index == 0 else "AI-generated"
        else:
            return jsonify({"error": "Unexpected model output shape"}), 500

        # Prepare the response
        response = {
            "predictions": [
                {
                    "label": label,
                    "confidence": confidence
                }
            ]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
