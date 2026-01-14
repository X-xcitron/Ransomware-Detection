import os
import numpy as np
from PIL import Image
import tensorflow as tf

# ---- Edit these three paths ----
IMAGE_PATH  = r"D:\test_samples\bg.jpg"  # the image you want to classify
MODEL_PATH  = r"D:\Ransomware-Detection-using-Deep-Learning-master\output\model_checkpoint.keras"
LABELS_FILE = r"D:\Ransomware-Detection-using-Deep-Learning-master\label.txt"
IMAGE_SIZE  = 224

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    return labels

def preprocess_image(img_path, image_size=IMAGE_SIZE):
    img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
    x = np.asarray(img).astype(np.float32)
    x = tf.image.per_image_standardization(x)  # same normalization as training
    x = np.expand_dims(x.numpy(), axis=0)
    return x

def main():
    print("predict.py started")
    # Basic checks
    for p, name in [(IMAGE_PATH, "IMAGE_PATH"), (MODEL_PATH, "MODEL_PATH"), (LABELS_FILE, "LABELS_FILE")]:
        if not os.path.exists(p):
            print(f"[ERROR] {name} not found: {p}")
            return

    # Optional: safer GPU memory behavior
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    labels = load_labels(LABELS_FILE)
    is_binary = (len(labels) == 2)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Preprocessing: {IMAGE_PATH}")
    x = preprocess_image(IMAGE_PATH, IMAGE_SIZE)

    print("Predicting...")
    y = model.predict(x, verbose=0)

    if is_binary:
        prob = float(y[0][0])
        pred_idx = 1 if prob >= 0.5 else 0
        pred_label = labels[pred_idx]
        print(f"Prediction: {pred_label} | prob_Ransomware={prob:.4f} prob_benign={(1.0 - prob):.4f}")
    else:
        pred_idx = int(np.argmax(y[0]))
        pred_label = labels[pred_idx]
        conf = float(y[0][pred_idx])
        print(f"Prediction: {pred_label} | confidence={conf:.4f}")
        # Top-3 breakdown (optional)
        top_k = sorted([(labels[i], float(y[0][i])) for i in range(len(labels))],
                       key=lambda t: t[1], reverse=True)[:3]
        print("Top-3:")
        for lbl, p in top_k:
            print(f"  {lbl}: {p:.3f}")


if __name__ == "__main__":
    main()
