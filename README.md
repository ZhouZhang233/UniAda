# UniAda
The offical implementation of "UniAda: Domain Unifying and Adapting Network for Generalizable Medical Image Segmentation"


# Envirenment
```shell
conda create -n uniada python=3.9
conda activate uniada
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

# data preprocessing
Download the [prostate datatset](https://liuquande.github.io/SAML/) 
```shell
python data_preprocess.py
```

# Training
Start training !
```shell
python train.py --data_dir path/to/data
```

# Testing
Let's test the trained model!
```shell
python test_prostate_tta.py
```

If you have any questions, please contact me! (zz_zhang@stu.scu.edu.cn)
