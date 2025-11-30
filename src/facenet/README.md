# FaceNet Face Recognition

Implementasi Face Recognition menggunakan **FaceNet (Inception-ResNet)** dengan pretrained weights dari VGGFace2.

## 🎯 Features

- ✅ **FaceNet (InceptionResnetV1)** - State-of-the-art face recognition
- ✅ **512-dimensional embeddings** - Compact dan efficient
- ✅ **MTCNN face detection** - Automatic face cropping dan alignment
- ✅ **Data augmentation** - Rotation, flip, brightness adjustment
- ✅ **Training pipeline** - Complete training dengan validation
- ✅ **Save/Load model** - Persistent storage
- ✅ **Easy prediction** - Single line prediction API

## 📦 Requirements

```bash
pip install torch torchvision
pip install facenet-pytorch
pip install pillow opencv-python
pip install scikit-learn matplotlib numpy
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from facenet import FaceNetModel

# Initialize model
model = FaceNetModel(device='cuda')

# Load dataset
embeddings, labels = model.load_dataset("./train", augment=True)

# Train classifier
history = model.train_classifier(
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-3
)

# Save model
model.save_model('./models')

# Predict
name, similarity = model.predict("test_image.jpg", threshold=0.6)
print(f"Predicted: {name} (Similarity: {similarity:.4f})")
```

### 2. Dataset Structure

```
train/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### 3. Using Jupyter Notebook

Buka `facenet_demo.ipynb` untuk interactive demo lengkap dengan visualisasi.

## 📊 Model Architecture

### FaceNet Pipeline

```
Input Image (any size)
    ↓
MTCNN Face Detection
    ↓
Cropped Face (160x160)
    ↓
InceptionResnetV1 (pretrained VGGFace2)
    ↓
512-dim Embeddings
    ↓
Classifier Head (FC layers)
    ↓
Prediction
```

### Classifier Head

```
Input: 512-dim embedding
    ↓
FC1: 512 → 256 (ReLU + Dropout 0.3)
    ↓
FC2: 256 → 128 (ReLU + Dropout 0.3)
    ↓
FC3: 128 → num_classes
    ↓
Output: Class probabilities
```

## 🎓 Training

### Hyperparameters

```python
model.train_classifier(
    num_epochs=20,          # Training epochs
    batch_size=32,          # Batch size
    learning_rate=1e-3,     # Learning rate
    validation_split=0.2    # 80-20 train-val split
)
```

### Training Output

```
🎓 Training classifier...
======================================================================
📊 Data split:
   Training: 205 samples
   Validation: 52 samples
   Classes: 68

Epoch    | Train Loss   | Train Acc    | Val Loss     | Val Acc      | Status         
==========================================================================================
1        | 3.2415       | 35.61        | 2.8934       | 42.31        |                
2        | 2.1234       | 52.68        | 1.9876       | 58.46        | ✅ BEST
...
20       | 0.1234       | 98.54        | 0.3456       | 94.23        | ✅ BEST
==========================================================================================

✅ Training completed!
   Best Epoch: 18
   Best Val Accuracy: 94.23%
```

## 📈 Performance

### Advantages vs ResNet50

| Metric | FaceNet | ResNet50 |
|--------|---------|----------|
| Embedding Size | 512 | 2048 |
| Pretrained on | VGGFace2 (faces) | ImageNet (objects) |
| Face-specific | ✅ Yes | ❌ No |
| Memory Usage | 75% less | Baseline |
| Inference Speed | ~2x faster | Baseline |
| Accuracy | Excellent | Good |

## 🔧 Advanced Usage

### Custom Data Augmentation

```python
# Enable augmentation during dataset loading
embeddings, labels = model.load_dataset(
    data_dir="./train",
    augment=True  # Applies rotation, flip, brightness
)
```

### Load Pretrained Model

```python
model = FaceNetModel(device='cuda')
model.load_model('./models/facenet_model_20241130_123456.pkl')

# Now ready for prediction
name, similarity = model.predict("new_image.jpg")
```

### Batch Prediction

```python
import os

test_dir = "./test"
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    name, similarity = model.predict(img_path)
    print(f"{img_file}: {name} ({similarity:.3f})")
```

## 📚 API Reference

### `FaceNetModel`

#### Methods:

**`__init__(device='cuda')`**
- Initialize FaceNet model
- Args: `device` - 'cuda' or 'cpu'

**`load_dataset(data_dir, augment=False)`**
- Load dataset from folder structure
- Args: 
  - `data_dir`: Path to dataset folder
  - `augment`: Enable data augmentation
- Returns: `(embeddings, labels)`

**`train_classifier(num_epochs, batch_size, learning_rate, validation_split)`**
- Train classifier on embeddings
- Returns: `history` dict with training metrics

**`predict(image_path, threshold=0.6)`**
- Predict identity from image
- Args:
  - `image_path`: Path to image
  - `threshold`: Similarity threshold
- Returns: `(predicted_name, similarity_score)`

**`save_model(save_dir='./models')`**
- Save model to disk
- Returns: `model_path`

**`load_model(model_path)`**
- Load model from disk
- Returns: `model_data` dict

## 🐛 Troubleshooting

### Issue: "No face detected"
**Solution:** 
- Check image quality
- Ensure face is visible and not too small
- Adjust MTCNN thresholds in `__init__`

### Issue: "CUDA out of memory"
**Solution:**
- Reduce `batch_size` in training
- Use CPU: `model = FaceNetModel(device='cpu')`
- Process images in smaller batches

### Issue: Low accuracy
**Solution:**
- Enable data augmentation: `augment=True`
- Increase training epochs
- Add more training data
- Check data quality and labeling

## 📖 References

- [FaceNet Paper](https://arxiv.org/abs/1503.03832) - Schroff et al., 2015
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - PyTorch implementation
- [VGGFace2 Dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) - 3.3M images

## 📄 License

MIT License

## 👥 Authors

Face Recognition System for Deep Learning Project

## 🙏 Acknowledgments

- Google Research - FaceNet architecture
- VGGFace2 - Training dataset
- timesler - facenet-pytorch implementation
