# PyTorch-Image-Models-Multi-Label-Classification

This repository is used for multi-label classification. <br>
The code is based on [pytorch-image-models by Ross Wightman](https://github.com/rwightman/pytorch-image-models). <br>
Thank Ross for his great work.

I downloaded his code on February 27, 2021. <br>
I think my multi-label classification code would be compatible with his latest version, but I didn’t check.

The main reference for multi-label classification is [this website](https://learnopencv.com/multi-label-image-classification-with-pytorch/). <br>
Thank Dmitry Retinskiy and Satya Mallick. <br>
For the purpose of understanding our context and the dataset, please spend 5 minutes on reading the link above, though you don’t need to read the specific code there. <br>
[Here is the link to download the images.](https://www.kaggle.com/paramaggarwal/fashion-product-images-small) <br>
Put all the images into ./fashion-product-images/images/.

In order to implement multi-label classification, I modify (add) the following files from Ross’ pytorch-image-models:
1.	 ./train.py
2.	 ./validate.py
3.	 ./timm/data/__init__.py
4.	 ./timm/data/dataset.py
5.	 ./timm/data/loader.py
6.	 ./timm/models/__init__.py
7.	 ./timm/models/efficientnet.py
8.	 ./timm/models/multi_label_model.py (add)

**In order to train your own dataset, you only need to modify the 1, 2, 4, 8 files.** <br>
Simply modify the code between the double dashed lines, or search color/gender/article, that’s the code/label that you need to change.

In terms of backbones, I only modified ./timm/models/efficientnet.py, I add an as_sequential_for_ML method. <br>
For other models, you need to define the as_sequential_for_ML method yourself within each class. It’s simply a part of the as_sequential method. <br>
We only need the backbone at this moment, so remove the last layers, for example classifier layer, from the as_sequential method. See forward_features method in each model class, then you would know which layers you need to remove, or how to define the as_sequential_for_ML method.

In addition, besides the multi-label classification functionality, I also add gradient centralization within AdamP optimizer. <br>
[Gradient centralization](https://github.com/Yonghongwei/Gradient-Centralization) is a simple technique and may improve the optimizer performance. <br>
No guarantee it will improve, but it is worth giving a try. <br>
To add gradient centralization, I modify (add) the following files:
1.	 ./timm/optim/adamp.py
2.	 ./timm/optim/centralization.py (add)
3.	 ./timm/optim/optim_factory.py <br>

Obviously, you can add gradient centralization within other optimizers as well.

Also, I updated ./timm/utils/summary.py so that we can output learning rate to summary.csv during training. <br>
Hence you could draw your learning rate together with loss and accuracy for the whole training process.

Here is a command example to start to train: <br>
```
./distributed_train.sh 1 ./fashion-product-images/ --model efficientnet_b2 -b 64 --sched cosine --epochs 50 --decay-epochs 2.4 --decay-rate .97 --opt adamp --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016 --pretrained  
```

And a command example to start to validate: <br>
```
python validate.py ./fashion-product-images/ --model efficientnet_b2 --checkpoint ./output/train/your_specific_folder/model_best.pth.tar -b 64  
```
 
 <br>
Please give a star if you find this repo helpful.

### License
This project is released under the Apache License, Version 2.0.

### Citation (BibTeX)
```
@misc{yrx2021multilabel,
  author = {YANG Ruixin},
  title = {PyTorch Image Models Multi-Label Classification},
  year = {2021},
  publisher = {GitHub}
}
```
