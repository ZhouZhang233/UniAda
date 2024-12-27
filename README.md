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

If you have any questions, please contact me! (zz_zhang95@163.com)

# Acknowledgement
Our implementation is heavily drived from [DoFE](https://github.com/emma-sjwang/Dofe). The prostate dataset is proposed in [SAML](https://arxiv.org/pdf/2007.02035) and the NPC dataset is proposed in [RobustNPC](https://www.sciencedirect.com/science/article/pii/S016781402300018X). Thanks to their great work.
```shell

@inproceedings{liu2020shape,
  title={Shape-aware meta-learning for generalizing prostate {MRI} segmentation to unseen domains},
  author={Liu, Quande and Dou, Qi and Heng, Pheng-Ann},
  booktitle={Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Interv. (MICCAI)},
  pages={475--485},
  year={2020},
  organization={Springer}
}
@article{LUO2023109480,
  title = {Deep learning-based accurate delineation of primary gross tumor volume of nasopharyngeal carcinoma on heterogeneous magnetic resonance imaging: A large-scale and multi-center study},
  journal = {Radiotherapy and Oncology},
  volume = {180},
  pages = {109480},
  year = {2023},
  issn = {0167-8140},
  author = {Xiangde Luo and Wenjun Liao and Yuan He and Fan Tang and Mengwan Wu and Yuanyuan Shen and Hui Huang and Tao Song and Kang Li and Shichuan Zhang and Shaoting Zhang and Guotai Wang},
}
```
