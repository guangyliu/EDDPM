from PIL import Image
import os

# Define the source and destination folders
source_folder = '../ffhq/images1024x1024/'
destination_folder = 'samples/ffhq_original_imgs/'
new_size = (128, 128)  # Example size, change as needed

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Resize each image
for filename in os.listdir(source_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(source_folder, filename)
        img = Image.open(img_path)
        img = img.resize(new_size, Image.ANTIALIAS)
        
        # Save to destination folder
        img.save(os.path.join(destination_folder, filename))

print("Resizing completed.")
