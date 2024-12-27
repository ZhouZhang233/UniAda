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

# Acknowledgement
Our implementation is heavily drived from [DoFE](https://github.com/emma-sjwang/Dofe). The fundus, prostate, and NPC dataset is proposed in [DoFE](https://ieeexplore.ieee.org/document/9163289), [SAML](https://arxiv.org/pdf/2007.02035), [RobustNPC](https://www.sciencedirect.com/science/article/pii/S016781402300018X), respectively. Thanks to their great work!
```shell
@article{wang2020dofe,
  title={Do{FE}: Domain-oriented feature embedding for generalizable fundus image segmentation on unseen datasets},
  author={Wang, Shujun and Yu, Lequan and Li, Kang and Yang, Xin and Fu, Chi-Wing and Heng, Pheng-Ann},
  journal={IEEE Trans. Med. Imag.},
  volume={39},
  number={12},
  pages={4237--4248},
  year={2020},
  publisher={IEEE}
}
@inproceedings{liu2020shape,
  title={Shape-aware meta-learning for generalizing prostate MRI segmentation to unseen domains},
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

# Contact
If you have any questions, please contact me! (zz_zhang95@163.com)
