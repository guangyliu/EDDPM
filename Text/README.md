This the implementation of the text experiments of the paper **Generating, Reconstructing, and Representing Discrete and Continuous Data: Generalized Encoding-Decoding Diffusion Probabilistic Models**. 


## Preparation
### Recommended Environment
Firstly, please check your CUDA version.
<!-- , you can simply check by typing:
```shell
 nvidia-smi|grep 'CUDA Version'
``` -->
Then please visit [pytorch official website](https://pytorch.org/get-started/previous-versions/) to find one compatible version. For example your CUDA Version is 11.3 (find one version with *cudatoolkit=11.3*), you may create a new conda enviroment (named *diled*) by:
```shell
conda create -n diled python==3.9.1 pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
where [*pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch*] is copied from the [pytorch official website](https://pytorch.org/get-started/previous-versions/).

Then activate *diled* and install the required packages by running:
```shell
conda activate diled
bash build_envs.sh
```
If you meet any probelm during installing apex, please ***carefully check whether you install the correct version of pytorch that compatible with your CUDA version***.



### Prepare Datasets and Classifiers
We follow the same data preparation and the same classifiers from [*LatentOps*](https://github.com/guangyliu/LatentOps?tab=readme-ov-file#prepare-datasets) and provide the same scripts here:
#### Datasets
Download and process the datasets by running the script:
```shell
bash download_datasets.sh
```
#### Download the Classifiers
Download and process the external classifiers by running the script:
```shell
bash download_classifiers.sh
```

### Training
```shell
cd code
bash train_joint_split_data_DDP_yelp.sh
```