

import streamlit as st
import os
import tempfile
from pathlib import Path
from your_code import *
import os

# --- Replace with your functions ---


st.set_page_config(page_title="Hydroponic Image Processor", layout="centered")
st.title("🌿 Hydroponic Image Processor")

st.markdown("Upload or specify folders to process your plant images.")

# Folder selection
input_folder = st.text_input("📂 Folder with input images")

# Processing options
process = st.radio("Choose a function to run:", ["Crop", "Timelapse", "Mask", "Growth"])

# Optional second folder (e.g., masks for growth)
mask_folder = None
if process == "Growth":
    mask_folder = st.text_input("📂 Folder with masks")

# Output folder
output_folder = st.text_input("💾 Folder to save results")

# Crop ROI inputs (only show if Crop selected)
roi = None
if process == "Crop":
    st.subheader("🖼️ Crop Settings")
    x = st.number_input("Crop X", min_value=0, value=0)
    y = st.number_input("Crop Y", min_value=0, value=0)
    w = st.number_input("Crop Width", min_value=1, value=300)
    h = st.number_input("Crop Height", min_value=1, value=300)
    roi = (x, y, w, h)

# Run the selected process
if st.button(f"Run {process}"):
    if not os.path.isdir(input_folder):
        st.error("❌ Please provide a valid input folder path.")
    elif not os.path.isdir(output_folder):
        st.error("❌ Please provide a valid output folder path.")
    elif process == "Growth" and not (mask_folder and os.path.isdir(mask_folder)):
        st.error("❌ Please provide a valid mask folder path.")
    else:
        with st.spinner(f"Running {process.lower()} on {input_folder}..."):
            try:
                if process == "Crop":
                    run_cropping(input_folder, output_folder)
                elif process == "Timelapse":
                    temp_path = os.path.join(tempfile.gettempdir(), "timelapse.mp4")
                    run_timelapse(input_folder, temp_path)
                    with open(temp_path, "rb") as f:
                        st.download_button("Download Timelapse", f, file_name="timelapse.mp4")
                elif process == "Mask":
                    run_mask(input_folder, output_folder)
                elif process == "Growth":
                    run_growth(input_folder, mask_folder, output_folder)
                st.success("✅ Done!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
