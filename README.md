# UniAda
The offical implementation of "UniAda: Domain Unifying and Adapting Network for Generalizable Medical Image Segmentation"


# Envirenment
```shell
conda create -n uniada python=3.9
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

# datat preprocessing
Download the [prostate datatset](https://liuquande.github.io/SAML/) 
```shell
python data_preprocess.py
```

# Training
train the model
```shell
python train.py
```

# Testing
```shell
python test_prostate_tta.py
```
