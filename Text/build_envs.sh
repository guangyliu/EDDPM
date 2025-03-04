
# Install required pip packages
pip install git+https://github.com/huggingface/transformers
pip install nltk==3.7 
pip install boto3 sacremoses tensorboardX torchdiffeq einops npy_append_array 
pip install pudb accelerate datasets
pip install -i https://testpypi.python.org/pypi peft

# Install Apex
rm -rf apex
git clone https://github.com/NVIDIA/apex
cd apex
#pip install -v --disable-pip-version-check --no-cache-dir ./
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

