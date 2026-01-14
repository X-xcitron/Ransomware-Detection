from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO
import os

app = Flask(__name__)

# ✅ YOUR EXACT PATHS
MODEL_PATH = r'..\output\model_checkpoint.keras'
LABELS_FILE = r'..\label.txt'





print("🔄 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_FILE, 'r') as f:
        labels = [line.strip() for line in f]
    print(f"✅ Model loaded! Classes: {labels}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

@app.route('/')
def home():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>🛡️ Ransomware Detection</title>
    <meta name="viewport" content="width=device-width">
    <style>
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:Arial;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;display:flex;align-items:center;justify-content:center;}
        .container{background:white;padding:2rem;border-radius:20px;box-shadow:0 20px 40px rgba(0,0,0,0.1);max-width:500px;width:90%;text-align:center;}
        h1{color:#333;margin-bottom:1rem;}
        .upload-area{border:3px dashed #667eea;border-radius:15px;padding:3rem;margin:2rem 0;cursor:pointer;transition:all 0.3s;background:#f8f9ff;}
        .upload-area:hover{border-color:#764ba2;background:#f0f2ff;}
        #preview{max-width:300px;max-height:300px;border-radius:10px;margin:1rem 0;display:none;}
        .btn{background:linear-gradient(45deg,#667eea,#764ba2);color:white;border:none;padding:15px 40px;border-radius:25px;font-size:1.1rem;cursor:pointer;transition:all 0.3s;}
        .btn:hover:not(:disabled){transform:translateY(-3px);}
        .btn:disabled{opacity:0.6;cursor:not-allowed;}
        .result{margin-top:2rem;padding:2rem;border-radius:15px;font-size:1.2rem;}
        .result.safe{background:#d4edda;color:#155724;border:3px solid #c3e6cb;}
        .result.malware{background:#f8d7da;color:#721c24;border:3px solid #f5c6cb;}
        .loading{display:none;margin:2rem 0;}
        .spinner{border:4px solid #f3f3f3;border-top:4px solid #667eea;border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:0 auto;}
        @keyframes spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
        .probs{text-align:left;background:rgba(255,255,255,0.9);padding:1.2rem;border-radius:10px;margin-top:1rem;font-size:0.95rem;}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Ransomware Detection</h1>
        <p style="color:#666;margin-bottom:2rem;">Upload image → AI Analysis → Instant Result</p>
        <div class="upload-area" id="uploadArea">
            <div style="font-size:1.2rem;margin-bottom:0.5rem;">📁 Click or Drag & Drop</div>
            <input type="file" id="imageInput" accept="image/*" style="display:none;">
        </div>
        <img id="preview" src="" alt="Preview">
        <br><button class="btn" id="predictBtn" onclick="predictImage()" disabled>🔍 Analyze</button>
        <div class="loading" id="loading"><div class="spinner"></div><div>Analyzing...</div></div>
        <div id="result" class="result" style="display:none;"></div>
    </div>
    <script>
        const uploadArea=document.getElementById('uploadArea'),imageInput=document.getElementById('imageInput'),preview=document.getElementById('preview'),predictBtn=document.getElementById('predictBtn'),loading=document.getElementById('loading'),result=document.getElementById('result');
        uploadArea.addEventListener('click',()=>imageInput.click());
        uploadArea.addEventListener('dragover',e=>{e.preventDefault();uploadArea.style.borderColor='#28a745';});
        uploadArea.addEventListener('dragleave',()=>{uploadArea.style.borderColor='#667eea';});
        uploadArea.addEventListener('drop',e=>{e.preventDefault();uploadArea.style.borderColor='#667eea';const files=e.dataTransfer.files[0];if(files){const dt=new DataTransfer();dt.items.add(files);imageInput.files=dt.files;displayPreview(files);}});
        imageInput.addEventListener('change',e=>{if(e.target.files[0])displayPreview(e.target.files[0]);});
        function displayPreview(file){const reader=new FileReader();reader.onload=e=>{preview.src=e.target.result;preview.style.display='block';predictBtn.disabled=false;};reader.readAsDataURL(file);}
        async function predictImage(){const file=imageInput.files[0];if(!file)return alert('Select image!');loading.style.display='block';predictBtn.disabled=true;result.style.display='none';const formData=new FormData();formData.append('image',file);try{const response=await fetch('/predict',{method:'POST',body:formData});const data=await response.json();if(data.error)alert('Error: '+data.error);else showResult(data);}catch(e){alert('Error: '+e.message);}finally{loading.style.display='none';predictBtn.disabled=false;}}
        function showResult(data){preview.src='data:image/jpeg;base64,'+data.image;const isMalware=data.prediction.toLowerCase().includes('malware');result.className=`result ${isMalware?'malware':'safe'}`;let probs='';for(let[label,prob]of Object.entries(data.probabilities)){probs+=`<div><strong>${label}:</strong> ${prob}</div>`;}result.innerHTML=`<div style="font-size:1.5rem;">${data.prediction}</div><div style="font-size:1.3rem;">Confidence: <strong>${data.confidence}</strong></div><div class="probs"><strong>Probabilities:</strong><br>${probs}</div>`;result.style.display='block';}
    </script>
</body>
</html>'''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        if file is None:
            return jsonify({'error': 'No file uploaded'}), 400

        # --- same preprocessing as your predict.py ---
        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        x = np.asarray(img).astype(np.float32)
        x = tf.image.per_image_standardization(x)
        x = np.expand_dims(x.numpy(), axis=0)

        # --- predict (binary: output is one number) ---
        y = model.predict(x, verbose=0)
        prob_malware = float(y[0][0])
        prob_benign = 1.0 - prob_malware

        # label.txt order should be: benign, malware
        predicted_class = labels[1] if prob_malware >= 0.5 else labels[0]
        confidence = max(prob_benign, prob_malware)

        # preview image as base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.1%}",
            'probabilities': {
                labels[0]: f"{prob_benign:.1%}",
                labels[1]: f"{prob_malware:.1%}",
            },
            'image': img_str
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

    