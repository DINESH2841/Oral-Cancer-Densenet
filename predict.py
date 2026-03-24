import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except ImportError:
    print("Error: TensorFlow not installed. Run: pip install tensorflow")
    exit(1)

def load_and_preprocess_image(image_path):
    print(f"Loading image from {image_path}...")
    try:
        # Expected input shape of DenseNet201 model implemented in train.py
        img = load_img(image_path, color_mode='rgb', target_size=(128, 128))
    except Exception as e:
        print(f"Failed to load image: {e}")
        exit(1)
        
    img = img_to_array(img)
    img = img / 255.0  # Scale
    img = np.expand_dims(img, axis=0)
    return img

def predict():
    parser = argparse.ArgumentParser(description="Predict Oral Cancer from Image")
    parser.add_argument("--image", required=True, help="Path to the cell microscopic or clinical image")
    parser.add_argument("--model", default="best_model.h5", help="Path to trained model (.h5)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Target image file {args.image} not found.")
        return

    if not os.path.exists(args.model):
        print(f"\n[!] Error: Model file '{args.model}' not found.")
        print("[!] Please run 'python train.py' to train and save the model, or specify correct path with --model.")
        return
        
    print("Loading AI Model (DenseNet201 + AO)...")
    try:
        model = tf.keras.models.load_model(args.model, compile=False)
    except Exception as e:
        print(f"Failed to load model: {e}")
        from sys import exit
        exit(1)

    img_tensor = load_and_preprocess_image(args.image)
    
    print("Running Inference...")
    predictions = model.predict(img_tensor, verbose=0)
    
    # Based on train.py mapping: {'Non-Cancer':0, 'Cancer':1} 
    # Output structure: 2 nodes (Softmax). Index 0 = Non-Cancer, Index 1 = Cancer
    pred_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][pred_idx] * 100
    
    classes = ["Non-Cancer", "Cancer"]
    result = classes[pred_idx]
    
    print("-" * 40)
    print(f"✅ Prediction : {result}")
    print(f"🔍 Confidence : {confidence:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    predict()
