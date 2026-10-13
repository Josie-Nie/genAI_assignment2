# CNN Image Classification – Assignment 2

## Overview
This project extends the previous FastAPI application by implementing a Convolutional Neural Network (CNN) using PyTorch for CIFAR-10 image classification.

## CNN Architecture
The CNN follows the required architecture:

- Input: 64 × 64 × 3 RGB image
- Conv2D (16 filters, 3×3) → ReLU → MaxPooling (2×2)
- Conv2D (32 filters, 3×3) → ReLU → MaxPooling (2×2)
- Flatten → Fully Connected (100 units) → ReLU
- Output: Fully Connected (10 classes)

CIFAR-10 images are resized from 32×32 to 64×64 before training and inference.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Model Training

The trained model is included at `models/cifar10_cnn.pth`. To retrain:

```bash
python train.py --epochs 10 --batch-size 64
```

## Run FastAPI

```bash
python -m uvicorn app.main:app --port 8000
```

API documentation: http://localhost:8000/docs

**API Endpoints:**
- `POST /generate` – Bigram text generation
- `POST /embedding` – spaCy word embeddings
- `POST /predict-image` – CNN image classification

## Docker Deployment

```bash
docker build -t genai-assignment2 .
docker run --rm -p 8000:80 genai-assignment2
```
