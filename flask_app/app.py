import boto3
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO  

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# AWS credentials 
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MODEL_PATH = "models/model_01.h5"  
LOCAL_MODEL_PATH = "/tmp/model_01.h5"  

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
        img_array = np.expand_dims(img_array, axis=0) / 255.0  

        # Make prediction
        preds = model.predict(img_array)  

        # Get predicted class
        confidence = preds[0]  
        label_index = np.argmax(confidence)  
        labels = ["Human-generated", "AI-generated"]  
        label = labels[label_index]

        # Format response
        response = {
            "label": label,
            "confidence": float(confidence[label_index])  
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500  

if __name__ == "__main__":
    app.run(debug=True)
