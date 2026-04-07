# 🩺 Skin Disease Classification App

A **Streamlit web application** for classifying skin lesions using a deep learning model trained on the **HAM10000 dataset**. The app features single image upload, batch processing from a sample folder, and detailed prediction confidence metrics.

---

## 📋 Table of Contents

- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Supported Classes](#-supported-classes)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## ✨ Features

✅ **Single Image Classification** - Upload and classify individual skin lesion images  
✅ **Batch Processing** - Process multiple images from the `samples/` folder  
✅ **Confidence Visualization** - View prediction confidence for all 7 classes  
✅ **Detailed Metrics** - See per-class probabilities with interactive charts  
✅ **Export Results** - Download predictions as text or CSV  
✅ **Responsive UI** - Clean, intuitive Streamlit interface  
✅ **GPU Support** - Automatic CUDA detection for faster inference  
✅ **Model Checkpoint Loading** - Configurable model path via sidebar  

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | EfficientNet-B3 (pretrained on ImageNet) |
| **Number of Classes** | 7 skin disease types |
| **Input Size** | 224×224 pixels |
| **Architecture** | Backbone + Custom Head with Dropout & BatchNorm |
| **Framework** | PyTorch + TIMM |

### Custom Head
```python
Sequential(
  BatchNorm1d(1536),
  Dropout(0.4),
  Linear(1536 → 512),
  ReLU(),
  BatchNorm1d(512),
  Dropout(0.2),
  Linear(512 → 7)
)
```

---

## 📥 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- GPU (optional, but recommended)

### Step 1: Clone or Download the Repository
```bash
git clone <your-repo-url>
cd skin-disease-classifier
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

### Step 4: Place Your Model Checkpoint
Download or copy your trained model checkpoint (`best_model.pt`) to the app directory:
```
.
├── app.py
├── best_model.pt          ← Place your model here
├── samples/               ← Add images here for batch processing
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── requirements.txt
```

---

## 🚀 Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Tab 1: 📸 Upload Image
1. Click "Browse files" to select a JPG or PNG image
2. The app will automatically:
   - Preprocess the image (resize to 224×224, normalize)
   - Run inference with the model
   - Display predictions with confidence scores
   - Show probability distribution across all classes

### Tab 2: 📁 Sample Folder
1. Create a `samples/` folder in your app directory
2. Add skin lesion images (.jpg or .png)
3. Click **"🚀 Process All Images"** to batch process
4. View results in a table and download as CSV

### Tab 3: ℹ️ About
Information about model architecture, training details, and dataset.

---

## 📁 Project Structure

```
skin-disease-classifier/
│
├── app.py                          # Main Streamlit application
├── best_model.pt                   # Trained model checkpoint (44 MB)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── samples/                        # (Optional) Sample images for batch processing
│   ├── sample1.jpg
│   ├── sample2.png
│   └── ...
│
└── skin_disease.py                 # Original training script (optional)
```

---

## 🎯 Supported Classes

The model classifies skin lesions into 7 categories:

| Code | Class | Description |
|------|-------|-------------|
| **akiec** | Actinic Keratosis | Solar keratosis; precancerous lesion |
| **bcc** | Basal Cell Carcinoma | Most common type of skin cancer |
| **bkl** | Benign Keratosis | Harmless, benign growths |
| **df** | Dermatofibroma | Fibrous tissue lesion (benign) |
| **mel** | Melanoma | Most serious type of skin cancer |
| **nv** | Melanocytic Nevus | Common moles (benign) |
| **vasc** | Vascular Lesion | Blood vessel abnormalities |

---

## 🌐 Deployment

### Deploy on Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch, and `app.py`
   - Click "Deploy"

### Deploy on Other Platforms

**Hugging Face Spaces:**
- Create a new Space with Streamlit SDK
- Upload `app.py`, `requirements.txt`, and `best_model.pt`
- HF Spaces supports up to 50 GB storage (sufficient for 44 MB model)

**AWS/Google Cloud/Azure:**
- Use Docker with Streamlit
- Example Dockerfile:
  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
  ```

---

## 📊 Training Details

### Dataset
- **HAM10000** - 10,000 dermatoscopic images
- **Classes:** 7 skin disease types (imbalanced)
- **Image Size:** 600×450 pixels (resized to 224×224 for model)

### Training Configuration
```python
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE (Early Stopping) = 6

Loss Function: CrossEntropyLoss (weighted + label smoothing)
Optimizer: AdamW
Scheduler: CosineAnnealingLR
```

### Data Augmentation
- Random horizontal & vertical flips
- Random rotation (±20°)
- Color jittering (brightness, contrast, saturation, hue)
- Random affine transforms (translation 10%)
- RandomErasing (Cutout) - probability 0.2

### Regularization
- Dropout: 0.4 (backbone) & 0.2 (head)
- Batch Normalization
- Label Smoothing: 0.1
- Gradient Clipping: 1.0

---

## ⚙️ Configuration

Edit the sidebar in `app.py` to customize:

```python
# In the sidebar:
model_path = st.sidebar.text_input(
    "Model Checkpoint Path",
    value="best_model.pt"
)
```

Or modify constants in the code:
```python
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 🖼️ Example Output

### Single Image Prediction
- Input image displayed
- Predicted class with confidence percentage
- Bar chart showing probabilities for all 7 classes
- Option to download results as text file

### Batch Processing
- Table with filename, prediction, and confidence
- Sample predictions with images
- Option to download results as CSV

---

## ⚠️ Important Disclaimer

**⚠️ MEDICAL DISCLAIMER**

This application is for **educational and research purposes only**. 

**DO NOT use this model for medical diagnosis or decision-making.**

- This model is trained on a limited dataset (10,000 images)
- AI predictions can be incorrect or misleading
- Always consult a **qualified dermatologist** for proper diagnosis
- Skin lesions require professional medical evaluation
- Never delay medical care based on app predictions

**By using this app, you agree that:**
- The creators are not liable for any misuse or medical consequences
- This tool should only be used for learning and research
- Professional medical advice supersedes any prediction

---

## 📈 Performance Metrics

After training on HAM10000:

| Metric | Value |
|--------|-------|
| **Validation AUC (Macro)** | ~0.95 |
| **Best Epoch** | ~15-20 |
| **Model Size** | 44 MB |
| **Inference Time** | ~50-100ms (GPU) / ~300ms (CPU) |

---

## 🔧 Troubleshooting

### Model Not Found
**Error:** "Model checkpoint not found"
- **Solution:** Ensure `best_model.pt` is in the same directory as `app.py`
- Or provide the correct path in the sidebar

### Out of Memory
- **Solution:** Reduce batch size or use CPU mode
- For Streamlit Cloud, the model should work fine (44 MB is within limits)

### Slow Predictions
- **Solution:** Ensure CUDA is available (`torch.cuda.is_available()` returns True)
- CPU inference is slower but still functional

### No Images in Sample Folder
- **Solution:** Create a `samples/` directory and add .jpg or .png files

---

## 📚 References

- **HAM10000 Dataset:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **EfficientNet Paper:** https://arxiv.org/abs/1905.11946
- **TIMM Library:** https://github.com/rwightman/pytorch-image-models
- **PyTorch:** https://pytorch.org
- **Streamlit:** https://streamlit.io

---

## 📝 License

This project is provided as-is for educational purposes. The HAM10000 dataset is publicly available and can be used under its respective license terms.

---

## 👨‍💻 Author

Created for skin disease classification research using deep learning.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

---

## 📧 Support

For issues, questions, or suggestions, please create an issue on GitHub or contact the maintainers.

---

**Last Updated:** 2024  
**Version:** 1.0  
**Status:** Production Ready ✅