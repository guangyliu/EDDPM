from PIL import Image
import os

def resize_images(source_folder, target_folder, size=(256, 256)):
    """
    Resizes all images in the source_folder to the specified size and saves them to the target_folder.
    """
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Loop through all files
    for file in files:
        # Construct the full file path
        source_file = os.path.join(source_folder, file)
        target_file = os.path.join(target_folder, file)
        
        # Open and resize the image
        with Image.open(source_file) as img:
            img_resized = img.resize(size, Image.ANTIALIAS)
            
            # Save the resized image to the target folder
            img_resized.save(target_file)

        print(f"Resized and saved: {file}")

# Define your source and target folders
source_folder = '../ffhq/images1024x1024'
target_folder = 'samples/original_imgs_256'

# Call the function
resize_images(source_folder, target_folder)
