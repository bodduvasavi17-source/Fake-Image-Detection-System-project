import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from ela import convert_to_ela
from orb import detect_copy_move

# --- Helper Functions ---

def check_metadata(uploaded_file):
    """Checks for camera EXIF data to distinguish between real photos and AI/Internet images."""
    try:
        img = Image.open(uploaded_file)
        exif_data = img._getexif()
        # Reset file pointer so it can be read again by other functions
        uploaded_file.seek(0)
        
        if not exif_data:
            return "No Camera Metadata Found (Common in AI/Internet images)", False
        return "Camera Metadata Detected (Likely a Real Photo)", True
    except Exception:
        return "Metadata analysis failed", False

@st.cache_resource
def get_model():
    """Loads the pre-trained H5 model."""
    return load_model("model.h5")

# --- Main Application ---

st.title("Fake Image Detection System")

# --- Sidebar Instructions ---
with st.sidebar:
    st.header("How to use")
    st.info("""
    1. **Upload Original Files**: Use the direct .jpg from your camera.
    2. **Avoid Screenshots**: Screenshots strip metadata and may trigger a 'Fake' result.
    3. **Check ELA**: Look for high-contrast areas in the ELA view; these indicate tampering.
    """)

model = get_model()
uploaded = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    # 1. Display Original Image
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 2. Metadata Check 
    metadata_msg, metadata_found = check_metadata(uploaded)
    st.info(metadata_msg)

    # 3. ELA Processing
    ela_img_array = convert_to_ela(uploaded, target_size=(128, 128))
    st.image(ela_img_array, caption="ELA Analysis View", use_container_width=True)

    # 4. CNN Prediction
    processed = np.reshape(ela_img_array, (1, 128, 128, 3))
    prediction = model.predict(processed)
    score = prediction[0][0]

    # 5. ORB Check (Indented correctly inside 'if uploaded')
    orb_score = detect_copy_move(uploaded)
    
    st.write(f"ORB Keypoint Matches: {orb_score}")
    if orb_score > 50:
        st.warning("⚠️ High number of matching keypoints detected. This may indicate a 'Copy-Move' forgery.")
    else:
        st.success("No significant Copy-Move patterns detected.")

    st.divider()

    # 6. Final Decision Logic
    if not metadata_found:
        # AI-consistent patterns often have very low/flat ELA scores
        if score < 0.05:
            st.error(f"🚩 Result: Likely a FAKE / AI Generated Image (Score: {score:.4f})")
            st.warning("Note: No camera fingerprint found and AI-consistent pixel patterns detected.")
        else:
            st.error(f"Result: Likely a FAKE / TAMPERED Image (Score: {score:.4f})")
    else:
        # If metadata is present, use standard threshold
        if score < 0.5:
            st.success(f"✅ Result: Likely a REAL Image (Score: {score:.4f})")
        else:
            st.error(f"Result: Likely a FAKE Image (Score: {score:.4f})")

    # --- Technical Deep Dive ---
    with st.expander("View Technical Analysis Details"):
        st.write(f"**Model Confidence Score:** {score:.6f}")
        st.write(f"**Metadata Status:** {'Found' if metadata_found else 'Missing'}")
        st.write("**Analysis Technique:** Convolutional Neural Network (CNN) + Error Level Analysis (ELA)")