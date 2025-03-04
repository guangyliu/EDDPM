import os
import sys
from PIL import Image
import numpy as np

def convert_png_to_npz(folder_path, output_file):
    # Initialize a list to hold the image arrays
    images = []

    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Open the image and resize it to 256x256
            with Image.open(file_path) as img:
                img = img.resize((256, 256))

                # Convert the image to a numpy array and append to the list
                images.append(np.asarray(img))

    # Stack all image arrays into a single numpy array
    all_images = np.stack(images)

    # Ensure the shape is (number, 256, 256, 3)
    # This step assumes all your images are RGB. If not, you might need to adjust.
    assert all_images.shape[1:] == (256, 256, 3), "Image shape is not correct."

    # Save the numpy array to an .npz file
    np.savez_compressed(output_file, all_images)

# Example usage
if len(sys.argv) > 1:
    folder_path = sys.argv[1]
    output_file = sys.argv[2]
convert_png_to_npz(folder_path, output_file)