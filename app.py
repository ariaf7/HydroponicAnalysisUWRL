
import io
import os
import tempfile
from io import BytesIO
from pathlib import Path
from your_code import *
import streamlit as st
import cv2
from your_code import run_timelapse, run_mask, run_growth
from PIL import Image
import numpy as np
import zipfile
import shutil
import uuid

st.set_page_config(page_title="Hydroponic Image Processor", layout="centered")
st.title("🌿 Hydroponic Image Processor")

st.markdown("Upload or specify folders to process your plant images.")

# Upload images
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Save uploaded files temporarily
temp_input_dir = os.path.join(tempfile.gettempdir(), "temp_uploaded_images")
os.makedirs(temp_input_dir, exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(temp_input_dir, file.name), "wb") as f:
            f.write(file.getbuffer())

# Operation selector
process = st.radio("Choose a function to run:", ["Timelapse", "Mask", "Growth"])

# Optional mask folder input for Growth operation
uploaded_masks = []
if process == "Growth":
    uploaded_masks = st.file_uploader(
        "Upload mask images (must contain 'mask' in filename)",
        type=["jpg", "png"],
        accept_multiple_files=True
    )
    # Filter only files that contain "mask" in their name
    uploaded_masks = [f for f in uploaded_masks if "mask" in f.name.lower()]


# Run the selected process
if st.button(f"Run {process}"):
    if not uploaded_files:
        st.error("❌ Please upload image files.")
    elif process == "Growth" and not uploaded_masks:
        st.error("❌ Please provide a valid mask folder path.")
    else:
        with st.spinner(f"Running {process.lower()}..."):
            try:
                if process == "Timelapse":
                    temp_path = os.path.join(tempfile.gettempdir(), "timelapse.mp4")
                    run_timelapse(temp_input_dir, temp_path)
                    with open(temp_path, "rb") as f:
                        st.download_button("Download Timelapse", f, file_name="timelapse.mp4")
                elif process == "Mask":
                    temp_output_dir = os.path.join(tempfile.gettempdir(), "mask_output")
                    os.makedirs(temp_output_dir, exist_ok=True)
                    run_mask(temp_input_dir, temp_output_dir)
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zipf:
                        for root, _, files in os.walk(temp_output_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_output_dir)  # keeps folder structure
                                zipf.write(file_path, arcname)
                    zip_buffer.seek(0)
                    st.download_button("Download Masked Images as ZIP", zip_buffer, file_name="masked_images.zip")
                    st.success("✅ Masking complete!")
                elif process == "Growth":
                    temp_mask_dir = os.path.join(tempfile.gettempdir(), "temp_uploaded_masks")
                    os.makedirs(temp_mask_dir, exist_ok=True)

                    # Save uploaded mask files
                    for file in uploaded_masks:
                        if "mask" in file.name.lower():
                            with open(os.path.join(temp_mask_dir, file.name), "wb") as f:
                                f.write(file.getbuffer())

                    temp_output_dir = os.path.join(tempfile.gettempdir(), "growth_output")
                    os.makedirs(temp_output_dir, exist_ok=True)
                    try:
                        run_growth(temp_input_dir, temp_mask_dir, temp_output_dir)
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Growth analysis failed: {e}")
                        st.text(traceback.format_exc())
                        raise

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zipf:
                        for file in os.listdir(temp_output_dir):
                            with open(os.path.join(temp_output_dir, file), "rb") as f:
                                zipf.writestr(file, f.read())
                    zip_buffer.seek(0)
                    st.download_button("Download Growth Results as ZIP", zip_buffer, file_name="growth_results.zip")
                    st.success("✅ Growth analysis complete!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
