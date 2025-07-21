import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from plantcv import plantcv as pcv
import streamlit as st
from types import SimpleNamespace

# --- Replace with your functions ---
def run_mask(folder, output_path):
    print(f"[MASK] Running on folder: {folder}")
    # Looping through all the images in the directory, since we're making a mask for every one
    count = 0
    output_folder = os.path.join(output_path, "masks")
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    for file in image_files:
        
    # Our workflow, don't worry about this, as well as input/output options

        args = SimpleNamespace(
            images=["test.jpg"],
            names="image1",
            result="lettuce_results",
            outdir=".",
            writeimg=True,
            debug="none",
            sample_label="genotype"
            )

        # Set debug to the global parameter 
        pcv.params.debug = args.debug

        
    # Read image in called "file", which "file" is our looping variable, which is the image we are currently looping by
        img = cv2.imread(file)

    # Convert the image to HSV color space to find our blacks and browns
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for brown color
        lower_brown = np.array([10, 100, 20])  
        upper_brown = np.array([20, 255, 200])
        
    # Define the range for black colors
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([360, 255, 50])
        lower_black2 = np.array([100, 59, 20])  
        upper_black2 = np.array([123, 140, 236])

    # Create masks for brown and black colors
        black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
        black2_mask = cv2.inRange(hsv_image, lower_black2, upper_black2)
        brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Combine the masks
    # Invert them because we want to get rid of everything these masks cover, not vice versa
        combined_mask = black2_mask| black_mask | brown_mask
        inverted_mask = cv2.bitwise_not(combined_mask)
        
    # Loop to take away some of the fuzziness in the mask, by eroding and dilating   
        kernel = np.ones((3, 3), np.uint8)
        for i in range(0, 4):
            if i == 0:
                eroded = cv2.erode(inverted_mask.copy(), kernel, iterations= i +1)
            else:
                eroded = cv2.erode(dilated.copy(), kernel, iterations= i+1)
            dilated = cv2.dilate(eroded.copy(), kernel, iterations= i +1) 
    # Now we want to save our perfected combined mask called eroded. 
    # The image path will be different, because the .split is specific to my path to take the date of the name 
        output_image_path = output_folder +  f"/mask{count}.png"

    # Save the mask
        cv2.imwrite(output_image_path, eroded)
        count +=1

    print(f"✅ Masking complete! Images saved to: {output_folder}")        

def run_growth(folder, mask_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    image_files = sorted(image_files)

    # Match masks either by index or by name (flexible)
    mask_files = sorted([f for f in glob.glob(os.path.join(mask_folder, '*')) if "mask" in os.path.basename(f).lower()])
    
    if len(image_files) != len(mask_files):
        raise ValueError(f"❌ Mismatch: {len(image_files)} images but {len(mask_files)} masks found.")

    for name, mask_path in zip(image_files, mask_files):
        print(f"📷 Processing: {name}")
        print(f"🎭 Using mask: {mask_path}")

        pcv.params.debug = "none"
        pcv.params.sample_label = "genotype"
        pcv.params.dpi = 100
        pcv.params.text_size = 10
        pcv.params.text_thickness = 20

        img = cv2.imread(name)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {name}")
            continue

        images_path_sort = [name]
        masks_path_sort = [mask_path]

        try:
            img0, _, _ = pcv.readimage(filename=next(Path(folder).rglob('*.png')))
        except StopIteration:
            print("❌ No .png files found.")
            return

        lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]
        th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        pcv.params.sample_label = "plant"
        th_fill = pcv.fill(bin_img=th, size=200)
        rois = pcv.roi.auto_grid(mask=th_fill, nrows=6, ncols=3, img=img0)
        valid_rois = rois.contours

        out = pcv.segment_image_series(images_path_sort, masks_path_sort, valid_rois, save_labels=True, ksize=3)
        most_recent_slice = out[:, :, -1]

        shape_img = pcv.analyze.size(img=img, labeled_mask=most_recent_slice, n_labels=18)
        shape_img = pcv.analyze.color(rgb_img=img, labeled_mask=most_recent_slice, n_labels=18, colorspaces="RGB")

        base_name = os.path.splitext(os.path.basename(name))[0]
        result_file = os.path.join(output_folder, f"lettuce_results_{base_name}.csv")
        pcv.outputs.save_results(filename=result_file, outformat="CSV")
        print(f"💾 Saved: {result_file}")
        count += 1

    # Aggregate CSVs
    dfs = []
    for file in glob.glob(os.path.join(output_folder, '*.csv')):
        df = pd.read_csv(file, delimiter=',')
        print(f"📂 Reading CSV: {file}")

        try:
            date = file.split('s_')[1].split('.')[0]
        except IndexError:
            date = 'unknown'
        df['date'] = pd.to_datetime(date, errors='coerce').dt.date
        df = df[(df['trait'] != 'red_frequencies') & (df['trait'] != 'blue_frequencies')]
        dfs.append(df)

    master_csv = os.path.join(output_folder, 'Master.csv')
    pd.concat(dfs).to_csv(master_csv, index=False)
    print(f"📊 Master CSV saved: {master_csv}")

    # Plotting
    df = pd.read_csv(master_csv)
    df_original = df.copy()

    for plant in df_original['sample'].unique():
        g_plot = sns.lineplot(
            data=df_original[(df_original['sample'] == plant) & (df_original['trait'] == 'green_frequencies')],
            x='label',
            y='value',
            hue='date'
        )
        plt.title(plant)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Green Frequencies')
        plt.ylabel('Percent of Pixels')
        plt.savefig(os.path.join(output_folder, f"{plant}_green_freqs.png"), bbox_inches='tight')
        plt.clf()

        df_sorted = df.sort_values(by='date')
        sns.lineplot(
            data=df_sorted[(df_sorted['sample'] == plant) & (df_sorted['trait'] == 'area')],
            x='date',
            y='value'
        )
        plt.title(f"{plant} area")
        plt.ylabel('Area in pixels')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_folder, f"{plant}_area.png"), bbox_inches='tight')
        plt.clf()

    print(f"✅ All plots saved to: {output_folder}")

def run_timelapse(folder, output_path):
    print(f"[TIMELAPSE] Running on folder: {folder}")
    
    img_array = []
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    for filename in image_files:
        img = cv2.imread(filename)
        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)

    if not img_array:
        print("⚠️ No valid images found in folder.")
        return

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, size)
    for img in img_array:
        out.write(img)
    out.release()

    print(f"✅ Timelapse saved to: {output_path}")

