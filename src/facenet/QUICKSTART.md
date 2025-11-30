# FaceNet Quick Start Guide

## 🚀 Cara Menggunakan FaceNet Face Recognition

### Method 1: Using Python Scripts (Recommended untuk Production)

#### 1. Training Model

```bash
cd src/facenet

# Basic training
python train.py --data_dir ../../train/train

# Training dengan augmentation
python train.py --data_dir ../../train/train --augment

# Custom hyperparameters
python train.py \
    --data_dir ../../train/train \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.001 \
    --augment \
    --device cuda
```

**Output:**
- Model akan tersimpan di `./models/facenet_model_TIMESTAMP.pkl`
- Classifier weights: `./models/facenet_classifier_TIMESTAMP.pth`

#### 2. Testing Model

```bash
# Test single image
python test.py \
    --model_path ./models/facenet_model_20241130_123456.pkl \
    --test_image ../../test/image1.jpg \
    --threshold 0.6

# Test all images in directory
python test.py \
    --model_path ./models/facenet_model_20241130_123456.pkl \
    --test_dir ../../test \
    --threshold 0.6
```

---

### Method 2: Using Jupyter Notebook (Recommended untuk Explorasi)

1. **Buka notebook:**
   ```bash
   jupyter notebook facenet_demo.ipynb
   ```

2. **Jalankan cells secara berurutan:**
   - Cell 1: Import dan initialize model
   - Cell 2: Load dataset
   - Cell 3: Train classifier
   - Cell 4: Visualisasi training history
   - Cell 5: Save model
   - Cell 6: Test single image
   - Cell 7: Test multiple images

---

### Method 3: Using Python API (Untuk Integration)

```python
from facenet import FaceNetModel

# 1. Initialize
model = FaceNetModel(device='cuda')

# 2. Load & Train
embeddings, labels = model.load_dataset("../../train/train", augment=True)
history = model.train_classifier(num_epochs=20, batch_size=32, learning_rate=1e-3)

# 3. Save
model_path = model.save_model('./models')

# 4. Predict
name, similarity = model.predict("../../test/image1.jpg", threshold=0.6)
print(f"Predicted: {name} (Similarity: {similarity:.4f})")
```

---

## 📂 Expected Folder Structure

```
Tubes/
├── train/
│   └── train/
│       ├── Person1/
│       │   ├── img1.jpg
│       │   └── img2.jpg
│       ├── Person2/
│       └── ...
├── test/
│   ├── test1.jpg
│   ├── test2.jpg
│   └── ...
└── src/
    └── facenet/
        ├── facenet.py
        ├── train.py
        ├── test.py
        ├── facenet_demo.ipynb
        └── models/
```

---

## ⚙️ Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `../../train/train` | Path ke training dataset |
| `--epochs` | `20` | Jumlah training epochs |
| `--batch_size` | `32` | Batch size untuk training |
| `--lr` | `0.001` | Learning rate |
| `--augment` | `False` | Enable data augmentation |
| `--device` | `cuda` | Device (cuda/cpu) |

### Testing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | Required | Path ke saved model |
| `--test_image` | `None` | Single image untuk test |
| `--test_dir` | `../../test` | Directory untuk batch test |
| `--threshold` | `0.6` | Similarity threshold |
| `--device` | `cuda` | Device (cuda/cpu) |

---

## 📊 Expected Output

### Training Output:
```
🚀 FaceNet Face Recognition Training
======================================================================

📦 Initializing FaceNet model...
🖥️  Device: cuda
✅ MTCNN loaded on CPU
✅ FaceNet (InceptionResnetV1) loaded on cuda
   Embedding dimension: 512

📂 Loading dataset from: ../../train/train
======================================================================
📂 Processing: Person1
📂 Processing: Person2
...
======================================================================
✅ Dataset loaded!
   Success: 257 faces
   Failed: 13 images
   Total embeddings: 1477 (with augmentation)
   Unique persons: 68

🎓 Training classifier...
======================================================================
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

💾 Saving model...
💾 Model saved!
   Embeddings: ./models/facenet_model_20241130_123456.pkl
   Classifier: ./models/facenet_classifier_20241130_123456.pth

======================================================================
✅ TRAINING COMPLETED!
======================================================================
```

### Testing Output:
```
🧪 FaceNet Face Recognition Testing
======================================================================

📦 Initializing FaceNet model...
📂 Loading model from: ./models/facenet_model_20241130_123456.pkl
✅ Model loaded from: ./models/facenet_model_20241130_123456.pkl

📂 Testing all images in: ../../test
✅ Found 15 test images

✅ image1.jpg                     → Abraham Ganda Napitu      (Sim: 0.856)
✅ image2.jpg                     → Bayu Ega Ferdana         (Sim: 0.782)
⚠️ image3.jpg                     → Unknown                  (Sim: 0.543)
...

======================================================================
📊 TEST SUMMARY
======================================================================
   Total images: 15
   Recognized: 12 (80.0%)
   Unknown: 2 (13.3%)
   No face: 1 (6.7%)
======================================================================
```

---

## 🎯 Tips untuk Hasil Optimal

1. **Data Quality:**
   - Minimal 3-5 gambar per person
   - Wajah jelas dan tidak tertutup
   - Variasi pose dan pencahayaan

2. **Training:**
   - Enable `--augment` untuk dataset kecil
   - Increase epochs jika validation accuracy masih naik
   - Monitor overfitting (train acc >> val acc)

3. **Testing:**
   - Adjust `threshold` berdasarkan use case:
     - `0.5-0.6`: Balanced (recommended)
     - `0.7-0.8`: High precision (lebih strict)
     - `0.4-0.5`: High recall (lebih toleran)

---

## 🐛 Common Issues & Solutions

### Issue: "CUDA out of memory"
```bash
# Solution 1: Reduce batch size
python train.py --batch_size 16

# Solution 2: Use CPU
python train.py --device cpu
```

### Issue: "No face detected" pada banyak gambar
```python
# Solution: Adjust MTCNN thresholds di facenet.py
self.mtcnn = MTCNN(
    thresholds=[0.3, 0.4, 0.4],  # Lebih toleran
    min_face_size=15  # Accept smaller faces
)
```

### Issue: Low validation accuracy
```bash
# Solution 1: Enable augmentation
python train.py --augment

# Solution 2: Increase epochs
python train.py --epochs 30

# Solution 3: Adjust learning rate
python train.py --lr 0.0005
```

---

## 📚 Next Steps

1. ✅ Train model dengan data Anda
2. ✅ Test dan evaluate performa
3. ✅ Adjust hyperparameters jika perlu
4. ✅ Deploy model untuk production
5. ✅ Monitor dan improve dari feedback

**Selamat mencoba! 🚀**
