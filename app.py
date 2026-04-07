import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import os
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLS = len(CLASSES)
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class descriptions
CLASS_INFO = {
    'akiec': 'Actinic Keratosis (Solar Keratosis)',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesion',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus (Mole)',
    'vasc': 'Vascular Lesion'
}

# ============================================================================
# Model Definition
# ============================================================================
class EfficientNetWithDropout(nn.Module):
    def __init__(self, num_classes, drop_rate=0.4):
        super().__init__()
        self.base = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        feat_dim = self.base.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.base(x))

# ============================================================================
# Load Model
# ============================================================================
@st.cache_resource
def load_model(model_path):
    """Load the trained model from checkpoint"""
    model = EfficientNetWithDropout(NUM_CLS).to(DEVICE)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        st.success(f"✓ Model loaded (Epoch {ckpt['epoch']} | AUC {ckpt['auc']:.4f})")
    else:
        st.warning("⚠ Model checkpoint not found. Using untrained model.")
    
    model.eval()
    return model

# ============================================================================
# Image Preprocessing
# ============================================================================
def preprocess_image(image):
    """Convert PIL image to model input tensor"""
    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return val_tfm(image).unsqueeze(0).to(DEVICE)

# ============================================================================
# Prediction
# ============================================================================
def predict(model, image):
    """Run inference on image"""
    tensor = preprocess_image(image)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx]
    
    return pred_class, confidence, probs

# ============================================================================
# UI Functions
# ============================================================================
def display_prediction_results(pred_class, confidence, probs, image):
    """Display prediction results with visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, width='stretch', caption="Input Image")
    
    with col2:
        st.metric(
            "🎯 Predicted Class",
            pred_class.upper(),
            f"{confidence*100:.1f}%"
        )
        st.write(f"**Description:** {CLASS_INFO[pred_class]}")
        st.write(f"**Confidence:** {confidence:.4f}")
    
    # Confidence bar chart
    st.subheader("📊 Class Probabilities")
    prob_data = {CLASSES[i]: probs[i] for i in range(NUM_CLS)}
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71' if CLASSES[i] == pred_class else '#3498db' for i in range(NUM_CLS)]
    bars = ax.barh(list(prob_data.keys()), list(prob_data.values()), color=colors)
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Confidence per Class')
    ax.set_xlim(0, 1)
    
    # Add percentage labels
    for i, (cls, prob) in enumerate(prob_data.items()):
        ax.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontweight='bold')
    
    st.pyplot(fig, width='stretch')

def load_sample_images():
    """Load images from sample folder"""
    sample_dir = Path('samples')
    
    if not sample_dir.exists():
        st.info("📁 Create a 'samples' folder in the app directory and add images to use this feature.")
        return []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = sorted([f for f in sample_dir.iterdir() if f.suffix in valid_extensions])
    
    if not images:
        st.info("📁 No images found in 'samples' folder. Add .jpg or .png images to get started.")
    
    return images

# ============================================================================
# Main App
# ============================================================================
def main():
    st.set_page_config(
        page_title="🩺 Skin Disease Classifier",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    model_path = st.sidebar.text_input(
        "Model Checkpoint Path",
        value="best_model.pt",
        help="Path to your trained model checkpoint"
    )
    
    # Load model
    model = load_model(model_path)
    
    # Main content
    st.title("🩺 Skin Disease Classification")
    st.markdown("""
    ### HAM10000 Dataset Classification
    This app classifies skin lesions into 7 categories using an **EfficientNet-B3** model trained on the HAM10000 dataset.
    
    **Supported Classes:**
    - 🔴 **Actinic Keratosis** (akiec) - Solar keratosis
    - 🔴 **Basal Cell Carcinoma** (bcc) - Skin cancer type
    - 🟡 **Benign Keratosis** (bkl) - Harmless growths
    - 🟠 **Dermatofibroma** (df) - Fibrous skin lesion
    - ⚫ **Melanoma** (mel) - Serious skin cancer
    - 🟢 **Melanocytic Nevus** (nv) - Common moles
    - 🔵 **Vascular Lesion** (vasc) - Blood vessel lesion
    """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📸 Upload Image", "📁 Sample Folder", "ℹ️ About"])
    
    # ========================================================================
    # Tab 1: Upload Image
    # ========================================================================
    with tab1:
        st.subheader("Upload a Skin Lesion Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG or PNG image of a skin lesion"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                
                with st.spinner("🔍 Analyzing image..."):
                    pred_class, confidence, probs = predict(model, image)
                
                display_prediction_results(pred_class, confidence, probs, image)
                
                # Download results
                st.download_button(
                    label="📥 Download Prediction",
                    data=f"Predicted Class: {pred_class}\nConfidence: {confidence:.4f}\n\nFull Probabilities:\n" +
                         "\n".join([f"{CLASSES[i]}: {probs[i]:.4f}" for i in range(NUM_CLS)]),
                    file_name=f"prediction_{pred_class}.txt",
                    mime="text/plain"
                )
            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
    
    # ========================================================================
    # Tab 2: Sample Folder
    # ========================================================================
    with tab2:
        st.subheader("📁 Batch Prediction from Sample Folder")
        st.markdown("""
        Add images to the `samples/` folder in your app directory to process them in batch.
        The app will automatically detect and classify all images.
        """)
        
        sample_images = load_sample_images()
        
        if sample_images:
            st.write(f"Found **{len(sample_images)}** images in samples folder")
            
            if st.button("🚀 Process All Images", width='stretch'):
                progress_bar = st.progress(0)
                results = []
                
                for idx, img_path in enumerate(sample_images):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        pred_class, confidence, probs = predict(model, image)
                        
                        results.append({
                            'Filename': img_path.name,
                            'Prediction': pred_class.upper(),
                            'Confidence': f'{confidence:.2%}',
                        })
                        
                        progress_bar.progress((idx + 1) / len(sample_images))
                    except Exception as e:
                        st.warning(f"⚠ Error processing {img_path.name}: {str(e)}")
                
                # Display results table
                if results:
                    st.subheader("📊 Batch Results")
                    st.dataframe(results, width='stretch')
                    
                    # Download CSV
                    import pandas as pd
                    df = pd.DataFrame(results)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                
                # Show first few images with predictions
                st.subheader("📸 Sample Predictions")
                for idx, img_path in enumerate(sample_images[:3]):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        pred_class, confidence, probs = predict(model, image)
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(image, width='stretch')
                        with col2:
                            st.write(f"**{img_path.name}**")
                            st.metric("Prediction", pred_class.upper(), f"{confidence*100:.1f}%")
                    except Exception as e:
                        st.warning(f"⚠ Error: {str(e)}")
    
    # ========================================================================
    # Tab 3: About
    # ========================================================================
    with tab3:
        st.subheader("ℹ️ About This Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Model Architecture
            - **Base Model:** EfficientNet-B3
            - **Classes:** 7 skin disease types
            - **Image Size:** 224×224 pixels
            - **Pretrained:** ImageNet
            
            ### Training Details
            - **Dataset:** HAM10000 (10,000+ images)
            - **Optimizer:** AdamW
            - **Loss:** CrossEntropyLoss (weighted + label smoothing)
            - **Scheduler:** Cosine Annealing
            """)
        
        with col2:
            st.markdown("""
            ### Data Augmentation
            - Random horizontal/vertical flips
            - Random rotation (±20°)
            - Color jittering
            - Random affine transforms
            - RandomErasing (Cutout)
            
            ### Regularization
            - Dropout (0.4 & 0.2)
            - Batch Normalization
            - Label Smoothing (0.1)
            - Gradient Clipping
            """)
        
        st.markdown("""
        ---
        ### ⚠️ Disclaimer
        This model is for **educational and research purposes only**. It should **NOT** be used for medical diagnosis.
        Always consult a qualified dermatologist for proper skin lesion evaluation and diagnosis.
        
        ### 📚 Reference
        **HAM10000 Dataset:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
        """)

if __name__ == "__main__":
    main()