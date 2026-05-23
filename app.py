import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield",
    page_icon="🛡️",
    layout="centered"
)

# ── Load Model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("models/best_model.pth", 
                                      map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ── Image Transform ────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ── Prediction Function ────────────────────────────────
def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)
        pred    = probs.argmax(1).item()
        confidence = probs[0][pred].item() * 100
    label = "Fake" if pred == 0 else "Real"
    return label, confidence, probs[0]

# ── UI ─────────────────────────────────────────────────
st.title("🛡️ DeepShield")
st.subheader("AI-Powered Deepfake Image Detector")
st.markdown("Upload a face image to check whether it is **Real** or **AI-Generated (Fake)**.")
st.divider()

uploaded_file = st.file_uploader("📤 Upload an Image", 
                                  type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            label, confidence, probs = predict(image)

        if label == "Fake":
            st.error(f"🚨 FAKE detected!")
            st.metric("Confidence", f"{confidence:.2f}%")
        else:
            st.success(f"✅ REAL image!")
            st.metric("Confidence", f"{confidence:.2f}%")

        st.divider()
        st.markdown("**Probability Breakdown:**")
        st.progress(float(probs[0]), text=f"Fake: {probs[0]*100:.2f}%")
        st.progress(float(probs[1]), text=f"Real: {probs[1]*100:.2f}%")

st.divider()
st.markdown("*Built with EfficientNet & PyTorch — DeepShield Project*")