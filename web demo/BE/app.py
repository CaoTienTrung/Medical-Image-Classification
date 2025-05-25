from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import os
from model.predict import predict
import warnings

name_map = {
    "normal": "Bình thường",
    "adenocarcinoma": "Ung thư biểu mô tuyến",
    "large.cell.carcinoma": "Ung thư biểu mô TB lớn",
    "squamous.cell.carcinoma": "Ung thư biểu mô TB vảy",
}

warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/predict", methods=["POST"])
def predict_image():
    try:
        # Get image data from request
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Save image temporarily
        image = Image.open(io.BytesIO(image_bytes))
        # Convert RGBA to RGB if necessary
        if image.mode == "RGBA":
            image = image.convert("RGB")
        temp_path = os.path.join(UPLOAD_FOLDER, "temp_image.jpg")
        image.save(temp_path)

        # Make prediction
        result = predict(temp_path, "model/configs.yaml")
        result = name_map[result]

        # Clean up temporary file
        os.remove(temp_path)

        return jsonify({"predictions": result, "status": "success"})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
