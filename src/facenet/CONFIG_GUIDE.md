# FaceNet Configuration Guide

## 📝 Cara Menggunakan

Sekarang Anda **tidak perlu menggunakan command line arguments** lagi!  
Cukup **edit nilai di dalam file** sesuai kebutuhan, lalu jalankan:

```bash
python train.py
python test.py
```

---

## 🎓 TRAIN.PY - Training Configuration

### Lokasi Edit: Bagian **CONFIGURATION** di dalam `train.py`

```python
# ==================== CONFIGURATION ====================
# Edit konfigurasi di sini sesuai kebutuhan

# Dataset paths
DATA_DIR = "../../train/train"          # Path ke training dataset
TEST_DIR = "../../test"                 # Path ke test dataset

# Training hyperparameters
NUM_EPOCHS = 20                         # Jumlah epoch (default: 20)
BATCH_SIZE = 32                         # Batch size (default: 32)
LEARNING_RATE = 1e-3                    # Learning rate (default: 0.001)
VALIDATION_SPLIT = 0.2                  # Train-val split (default: 0.2 = 20% validation)

# Augmentation
ENABLE_AUGMENTATION = True              # True = enable augmentation, False = disable

# Device
DEVICE = 'cuda'                         # 'cuda' untuk GPU, 'cpu' untuk CPU
```

### 📊 Contoh Konfigurasi

#### Training Cepat (Testing):
```python
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
ENABLE_AUGMENTATION = False
```

#### Training Standard (Recommended):
```python
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ENABLE_AUGMENTATION = True
```

#### Training Advanced (High Accuracy):
```python
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
ENABLE_AUGMENTATION = True
```

#### Training di CPU (Tanpa GPU):
```python
NUM_EPOCHS = 20
BATCH_SIZE = 16          # Lebih kecil untuk CPU
LEARNING_RATE = 1e-3
DEVICE = 'cpu'
```

---

## 🧪 TEST.PY - Testing Configuration

### Lokasi Edit: Bagian **CONFIGURATION** di dalam `test.py`

```python
# ==================== CONFIGURATION ====================
# Edit konfigurasi di sini sesuai kebutuhan

# Model path
MODEL_PATH = "./models/facenet_model_20241130_123456.pkl"  # Path ke saved model

# Test mode: pilih salah satu
TEST_MODE = "directory"                 # "single" untuk 1 image, "directory" untuk folder

# Single image test (jika TEST_MODE = "single")
TEST_IMAGE = "../../test/1.jpg"         # Path ke single test image

# Directory test (jika TEST_MODE = "directory")
TEST_DIR = "../../test"                 # Path ke test directory

# Threshold
SIMILARITY_THRESHOLD = 0.6              # Threshold untuk classification (0.0 - 1.0)

# Device
DEVICE = 'cuda'                         # 'cuda' untuk GPU, 'cpu' untuk CPU
```

### 📊 Contoh Konfigurasi

#### Test Single Image:
```python
MODEL_PATH = "./models/facenet_model_20241130_123456.pkl"
TEST_MODE = "single"
TEST_IMAGE = "../../test/person1.jpg"
SIMILARITY_THRESHOLD = 0.6
```

#### Test All Images (Batch):
```python
MODEL_PATH = "./models/facenet_model_20241130_123456.pkl"
TEST_MODE = "directory"
TEST_DIR = "../../test"
SIMILARITY_THRESHOLD = 0.6
```

#### Test dengan Threshold Ketat (High Precision):
```python
SIMILARITY_THRESHOLD = 0.75           # Lebih strict, fewer false positives
```

#### Test dengan Threshold Longgar (High Recall):
```python
SIMILARITY_THRESHOLD = 0.45           # Lebih toleran, fewer unknowns
```

---

## 🔧 Tips Konfigurasi

### Learning Rate:
- **1e-3 (0.001)** - Standard, paling umum digunakan
- **5e-4 (0.0005)** - Lebih smooth, untuk fine-tuning
- **1e-4 (0.0001)** - Sangat hati-hati, untuk dataset kecil

### Batch Size:
- **16** - Untuk GPU memory terbatas / CPU
- **32** - Standard, balanced
- **64** - Untuk GPU besar, training lebih cepat
- **128** - Untuk dataset sangat besar dengan GPU powerful

### Epochs:
- **10** - Quick test
- **20** - Standard training
- **30-50** - High accuracy training
- **100+** - Research purposes (watch for overfitting!)

### Similarity Threshold:
- **0.4-0.5** - Sangat toleran, banyak match (high recall)
- **0.6** - Balanced (recommended)
- **0.7-0.8** - Strict, hanya match yang sangat yakin (high precision)
- **0.9+** - Extremely strict, hampir identical

---

## 📂 Path Examples

### Absolute Path:
```python
DATA_DIR = "D:/deeplearning/Tubes/train/train"
TEST_DIR = "D:/deeplearning/Tubes/test"
MODEL_PATH = "D:/deeplearning/Tubes/src/facenet/models/facenet_model_xxx.pkl"
```

### Relative Path (dari src/facenet/):
```python
DATA_DIR = "../../train/train"         # Naik 2 level, masuk train/train
TEST_DIR = "../../test"                # Naik 2 level, masuk test
MODEL_PATH = "./models/facenet_model_xxx.pkl"  # Folder models di current dir
```

---

## 🚀 Quick Examples

### Example 1: Training dengan Augmentation
```python
# Edit train.py:
NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ENABLE_AUGMENTATION = True
DEVICE = 'cuda'

# Lalu jalankan:
python train.py
```

### Example 2: Test Single Image
```python
# Edit test.py:
MODEL_PATH = "./models/facenet_model_20241130_162409.pkl"
TEST_MODE = "single"
TEST_IMAGE = "../../test/abraham.jpg"
SIMILARITY_THRESHOLD = 0.6

# Lalu jalankan:
python test.py
```

### Example 3: Batch Test All Images
```python
# Edit test.py:
MODEL_PATH = "./models/facenet_model_20241130_162409.pkl"
TEST_MODE = "directory"
TEST_DIR = "../../test"
SIMILARITY_THRESHOLD = 0.6

# Lalu jalankan:
python test.py
```

---

## ✅ Workflow Lengkap

1. **Edit `train.py`** → Set DATA_DIR, NUM_EPOCHS, dll
2. **Run training:** `python train.py`
3. **Lihat output:** Note model filename (contoh: `facenet_model_20241130_162409.pkl`)
4. **Edit `test.py`** → Set MODEL_PATH dengan filename dari step 3
5. **Run testing:** `python test.py`
6. **Lihat hasil** → Accuracy, predictions, dll

---

## 🐛 Troubleshooting

### Kalau path salah:
```
❌ FileNotFoundError: [Errno 2] No such file or directory: '../../train/train'
```
**Fix:** Cek path relatif Anda atau gunakan absolute path

### Kalau model tidak ditemukan:
```
❌ FileNotFoundError: [Errno 2] No such file or directory: './models/facenet_xxx.pkl'
```
**Fix:** Pastikan MODEL_PATH sesuai dengan output dari train.py

### Kalau CUDA memory habis:
```
❌ RuntimeError: CUDA out of memory
```
**Fix:** 
```python
BATCH_SIZE = 16    # Kurangi batch size
# atau
DEVICE = 'cpu'     # Gunakan CPU
```

---

**Selamat mencoba! 🎉**
