

import streamlit as st
import os
import tempfile
from pathlib import Path
from your_code import *
import os

# --- Replace with your functions ---


import streamlit as st
import cv2
from pathlib import Path
from your_code import run_cropping, run_timelapse, run_mask, run_growth
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="Hydroponic Image Processor", layout="centered")
st.title("🌿 Hydroponic Image Processor")

input_folder = st.text_input("📂 Folder with input images")
output_folder = st.text_input("💾 Folder to save results")

process = st.radio("Choose a function to run:", ["Crop", "Timelapse", "Mask", "Growth"])

mask_folder = None
if process == "Growth":
    mask_folder = st.text_input("📂 Folder with masks")

def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

if input_folder and output_folder:
    image_files = list(Path(input_folder).rglob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

    if len(image_files) == 0:
        st.warning("No images found in input folder")
    else:
        if process == "Crop":
            img = load_image(image_files[0])
            height, width, _ = img.shape

            st.image(img, caption="Original Image")

            x = st.slider("X", 0, width-1, 0)
            y = st.slider("Y", 0, height-1, 0)
            w = st.slider("Width", 1, width - x, width//3)
            h = st.slider("Height", 1, height - y, height//3)

            cropped_img = img[y:y+h, x:x+w]
            st.image(cropped_img, caption="Cropped Preview")

            if st.button("Run Crop"):
                run_cropping(input_folder, output_folder, (x, y, w, h))
                st.success(f"✅ Cropped images saved to {output_folder}")

        else:
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
                            if process == "Timelapse":
                                import tempfile
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
