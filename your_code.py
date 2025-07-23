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
import traceback


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
    csv_output_dir = os.path.join(output_folder, "csv_temp")
    os.makedirs(csv_output_dir, exist_ok=True)
    try:
        print("🔁 Starting run_growth")
        count = 0
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        print(f"🖼 Found {len(image_files)} image(s)")

        for name in image_files:
            print(f"\n📷 Processing image: {name}")
            pcv.params.debug = "none"
            pcv.params.sample_label = "genotype"
            pcv.params.dpi = 100
            pcv.params.text_size = 10
            pcv.params.text_thickness = 20

            img = cv2.imread(name)
            if img is None:
                print(f"⚠️ Could not read image: {name}")
                continue

            # Mask filename assumption
            mask_name = os.path.join(mask_folder, f"mask{count}.png")
            if not os.path.exists(mask_name):
                print(f"⚠️ Missing mask file: {mask_name}")
                continue

            images_path_sort = sorted([name])
            masks_path_sort = sorted([mask_name])

            try:
                img0, _, _ = pcv.readimage(filename=next(Path(folder).rglob('*.png')))
            except StopIteration:
                print("❌ No .png files found in folder.")
                return

            lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
            a_channel = lab[:, :, 1]
            th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            pcv.params.sample_label = "plant"
            th_fill = pcv.fill(bin_img=th, size=200)
            rois = pcv.roi.auto_grid(mask=th_fill, nrows=6, ncols=3, img=img0)
            valid_rois = rois.contours

            print("📊 Segmenting image series...")
            out = pcv.segment_image_series(images_path_sort, masks_path_sort, valid_rois, save_labels=True, ksize=3)
            most_recent_slice = out[:, :, -1]

            shape_img = pcv.analyze.size(img=img, labeled_mask=most_recent_slice, n_labels=18)
            shape_img = pcv.analyze.color(rgb_img=img, labeled_mask=most_recent_slice, n_labels=18, colorspaces="RGB")

            out_csv = os.path.join(csv_output_dir, f"lettuce_results_{Path(name).stem}.csv")
            print(f"💾 Saving CSV to: {out_csv}")
            pcv.outputs.save_results(filename=out_csv, outformat="CSV")
            count += 1

        # Compile CSVs
        print("\n📁 Compiling CSV files...")
        dfs = []
        input_directory2 = csv_output_dir
        for file in glob.glob(os.path.join(input_directory2, '*.csv')):
            df = pd.read_csv(file, delimiter=',')
            print(f"📄 Reading: {file}")
            try:
                date = file.split('s_')[1].split('.')[0]
            except IndexError:
                print(f"⚠️ Could not extract date from: {file}")
                continue

            df['date'] = pd.to_datetime(date).date()
            df = df[(df['trait'] != 'red_frequencies') & (df['trait'] != 'blue_frequencies')]
            dfs.append(df)

        # Create and save master CSV
        master_csv_path = os.path.join(output_folder, 'Master.csv')
        pd.concat(dfs).to_csv(master_csv_path, index=False)
        print(f"✅ Saved master CSV to {master_csv_path}")

        # Read it back in for plotting
        print("📈 Generating plots...")
        df = pd.read_csv(master_csv_path)

        df_original = df.copy()

        # Create subfolders for plots
        green_plot_dir = os.path.join(output_folder, "green_freqs")
        area_plot_dir = os.path.join(output_folder, "area_plots")
        os.makedirs(green_plot_dir, exist_ok=True)
        os.makedirs(area_plot_dir, exist_ok=True)

        # Loop through plants and create plots
        for plant in df_original['sample'].unique():
            # GREEN FREQS PLOT
            sns.lineplot(
                data=df_original[(df_original['sample'] == plant) & (df_original['trait'] == 'green_frequencies')],
                x='label',
                y='value',
                hue='date'
            )
            plt.title(plant)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('Green Frequencies')
            plt.ylabel('Percent of Pixels')
            green_path = os.path.join(green_plot_dir, f"{plant}_green_freqs.png")
            plt.savefig(green_path, bbox_inches='tight')
            plt.clf()

            # AREA PLOT
            df = df.sort_values(by='date')
            sns.lineplot(
                data=df[(df['sample'] == plant) & (df['trait'] == 'area')],
                x='date',
                y='value'
            )
            plt.title(f"{plant} area")
            plt.ylabel("Area in pixels")
            plt.xticks(rotation=45)
            area_path = os.path.join(area_plot_dir, f"{plant}_area.png")
            plt.savefig(area_path, bbox_inches='tight')
            plt.clf()

        print("✅ Growth analysis complete!")
    except Exception as e:
        print("❌ Exception occurred in run_growth:")
        traceback.print_exc()
        raise


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

