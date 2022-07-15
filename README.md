# GIIDA
Code release for "GIIDA: Unsupervised Domain Adaptation via
Gradual Interpolation Intermediate Domain Auxiliary
" 

## Prerequisites
- torch>=1.7.0
- torchvision

## Training

VisDA-2017
```
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --per-class-eval --log logs/GIIDA/VisDA2017/ --trade-off1 1.0 --visual-T --img-path GIIDA_VisDA2017 -i 1000 --early 20 --mu 1
```

Office Home
```
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Ar -t Cl -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Ar2Cl --visual-T --img-path GIIDA_Ar2Cl -i 1000 --early 20
```

Office31
```
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office31 -d Office31 -s D -t A -a resnet50  --log logs/GIIDA/Office31/Office31_D2A --visual-T --img-path GIIDA_D2A -i 500 --early 25
```



## Acknowledgement
This code is heavily borrowed from  [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library/), [SSL](https://github.com/YBZh/Bridging_UDA_SSL), and [CST]( https://github.com/Liuhong99/CST). It is our pleasure to acknowledge their contributions.

