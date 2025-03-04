import zipfile

# for T in [10, 20, 50]:

#     # Define the names of the images
#     image_names = [f"samples/celeba_reconstructions_best_fid_ema_T={T}_N=50000/images/image_{i}.png" for i in range(500)]  # image_0.png to image_50.png

#     # Create a new zip file
#     with zipfile.ZipFile(f'images_{T}.zip', 'w') as zipf:
#         for image_name in image_names:
#             zipf.write(image_name)


# for T in [10, 20, 50]:

#     # Define the names of the images
#     image_names = [f"samples/interpolation_ddpm_T={T}_N=50000/batch_{i}/interpolation_0.2_0.9.png" for i in range(500)]  # image_0.png to image_50.png

#     # Create a new zip file
#     with zipfile.ZipFile(f'images_{T}.zip', 'w') as zipf:
#         for idx, image_name in enumerate(image_names):
#             zipf.write(image_name, arcname=os.path.basename(image_name).replace("_0.2_0.9", f"_{idx}"))

# Define the names of the images
image_names = [f"samples/original_imgs/original_image_{i}.png" for i in range(100)]  # image_0.png to image_50.png

# Create a new zip file
with zipfile.ZipFile(f'ffhq_images_original.zip', 'w') as zipf:
    for image_name in image_names:
        zipf.write(image_name)