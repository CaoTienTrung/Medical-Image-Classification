# Medical Image Classification Backend

This is a simple Flask backend that simulates receiving medical images and processing them through an ML model.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /predict

Receives a base64-encoded image and returns classification results.

**Request Body:**

```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**

```json
{
  "prediction": "class_name",
  "probabilities": {
    "Normal": 0.3,
    "Pneumonia": 0.4,
    "COVID-19": 0.3
  }
}
```

## Note

This is a simulation that returns random predictions. In a real application, you would replace the `SimulatedMLModel` class with your actual ML model implementation.
