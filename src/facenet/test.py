"""
Simple testing script untuk FaceNet Face Recognition
"""

from facenet import FaceNetModel
import os
import glob

def main():
    # ==================== CONFIGURATION ====================
    # Edit konfigurasi di sini sesuai kebutuhan
    
    # Model path
    MODEL_PATH = "./models\facenet_model_20251130_165355.pkl"
    
    # Test mode
    TEST_MODE = "directory"
    
    # Single image test (jika TEST_MODE = "single")
    TEST_IMAGE = "../../test/1.jpg"
    
    # Directory test (jika TEST_MODE = "directory")
    TEST_DIR = "../../test"
    
    # Threshold (0-1)
    SIMILARITY_THRESHOLD = 0.6
    
    # Device
    DEVICE = 'cuda'
    
    # ======================================================
    
    print("=" * 70)
    print("🧪 FaceNet Face Recognition Testing")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Test Mode: {TEST_MODE}")
    print(f"   Threshold: {SIMILARITY_THRESHOLD}")
    print(f"   Device: {DEVICE}")
    
    # Initialize model
    print("\n📦 Initializing FaceNet model...")
    model = FaceNetModel(device=DEVICE)
    
    # Load model
    print(f"\n📂 Loading model from: {MODEL_PATH}")
    model.load_model(MODEL_PATH)
    
    # Test mode
    if TEST_MODE == "single":
        # Single image test
        print(f"\n🖼️  Testing single image: {TEST_IMAGE}")
        predicted_name, similarity = model.predict(TEST_IMAGE, threshold=SIMILARITY_THRESHOLD)
        
        print("\n" + "=" * 70)
        print("🎯 PREDICTION RESULT")
        print("=" * 70)
        print(f"   Predicted Name: {predicted_name}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Threshold: {SIMILARITY_THRESHOLD}")
        print("=" * 70)
        
    elif TEST_MODE == "directory":
        # Directory test
        print(f"\n📂 Testing all images in: {TEST_DIR}")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        test_images = []
        for ext in image_extensions:
            test_images.extend(glob.glob(os.path.join(TEST_DIR, ext)))
        
        if not test_images:
            print("❌ No images found in test directory!")
            return
        
        print(f"✅ Found {len(test_images)} test images\n")
        
        # Test each image
        results = []
        correct = 0
        unknown = 0
        no_face = 0
        
        for img_path in test_images:
            img_name = os.path.basename(img_path)
            predicted_name, similarity = model.predict(img_path, threshold=SIMILARITY_THRESHOLD)
            
            results.append({
                'file': img_name,
                'prediction': predicted_name,
                'similarity': similarity
            })
            
            # Statistics
            if predicted_name == "No Face Detected":
                no_face += 1
                status = "❌"
            elif predicted_name == "Unknown":
                unknown += 1
                status = "⚠️"
            else:
                correct += 1
                status = "✅"
            
            print(f"{status} {img_name[:30]:<30} → {predicted_name:<25} (Sim: {similarity:.3f})")
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        print(f"   Total images: {len(test_images)}")
        print(f"   Recognized: {correct} ({correct/len(test_images)*100:.1f}%)")
        print(f"   Unknown: {unknown} ({unknown/len(test_images)*100:.1f}%)")
        print(f"   No face: {no_face} ({no_face/len(test_images)*100:.1f}%)")
        print("=" * 70)
    
    else:
        print(f"\n❌ Invalid TEST_MODE: {TEST_MODE}")
        print("   Valid options: 'single' or 'directory'")

if __name__ == "__main__":
    main()
