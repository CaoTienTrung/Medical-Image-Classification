from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)


class SimulatedMLModel:
    def __init__(self):
        self.classes = ["Normal", "Pneumonia", "COVID-19"]
        self.threshold = 0.5

    def predict(self, image):
        # Giả lập dự đoán với danh sách các class và confidence
        return {
            "predictions": [
                {"label": "Normal", "confidence": 0.7},
                {"label": "Pneumonia", "confidence": 0.2},
                {"label": "COVID-19", "confidence": 0.1},
            ]
        }


ml_model = SimulatedMLModel()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 string trực tiếp (FE đã bỏ prefix)
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data))

        # Gọi model để dự đoán
        result = ml_model.predict(image)

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
