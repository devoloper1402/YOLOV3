import os
import pandas as pd


def create_csv(image_dir, label_dir, output_csv):
    # Get sorted lists of image and label files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    # Check if the number of images and labels match
    if len(image_files) != len(label_files):
        print("Warning: The number of images and labels does not match.")

    # Create a list to store the rows for the CSV
    data = []

    # Iterate over image files and find corresponding label files
    for image_file in image_files:
        label_file = image_file.rsplit('.', 1)[0] + '.txt'  # Assume label has the same name with .txt extension
        if label_file in label_files:
            # Append row with image and label
            data.append({'image': image_file, 'label': label_file})
        else:
            print(f"Warning: No matching label found for {image_file}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")


# Specify directories and output CSV file name
image_dir = "C://Users//tgmad//PycharmProjects//YoloScratch//brain_test//images"
label_dir = "C://Users//tgmad//PycharmProjects//YoloScratch//brain_test//labels"
output_csv = "test.csv"

# Create the CSV file
create_csv(image_dir, label_dir, output_csv)
