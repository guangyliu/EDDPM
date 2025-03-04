This the implementation of the image experiments of the paper **Generating, Reconstructing, and Representing Discrete and Continuous Data: Generalized Diffusion with Learnable Encoding-Decoding**. 

To run the joint training, use the following command: 
```
python run_ffhq128_joint.py # for FFHQ dataset
python run_bedroom128_joint.py # for bedroom dataset
python run_horse128_joint.py # for horse dataset
python run_celeba64_joint.py # for celeba64 dataset
```

To run the evaluations, first copy the images from lmdb file to png files:
```
python copy_imgs.py
python copy_imgs_256.py
```

To run the evaluations of the generation of consistency models:
```
python cm_sample.py
python fid.py "samples/consistency_model_twosteps/samples_lpips/*.png" "samples/bedroom/original_imgs/*.png"
python cm_sample_256.py
python fid.py "samples/consistency_model_twosteps/samples_lpips_256/*.png" "samples/bedroom/original_imgs_256/*.png"
```

To run the evaluations of the reconstruction and interpolation of consistency models:
```
python cm_interpolate.py

# Evaluate reconstruction:
python fid.py "samples/consistency_model_twosteps/interpolation/0.0/*.png" "samples/bedroom/original_imgs/*.png"

# Evaluate alpha=0.2:
python fid.py "samples/consistency_model_twosteps/interpolation/0.2/*.png" "samples/bedroom/original_imgs/*.png"

# Evaluate alpha=0.4:
python fid.py "samples/consistency_model_twosteps/interpolation/0.4/*.png" "samples/bedroom/original_imgs/*.png"
```

