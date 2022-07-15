#!/usr/bin/env bashclear

### office31
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office31 -d Office31 -s D -t A -a resnet50  --log logs/GIIDA/Office31/Office31_D2A --visual-T --img-path GIIDA_D2A -i 500 --early 25
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office31 -d Office31 -s W -t A -a resnet50  --log logs/GIIDA/Office31/Office31_W2A --visual-T --img-path GIIDA_W2A -i 500 --early 25
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office31 -d Office31 -s A -t W -a resnet50  --log logs/GIIDA/Office31/Office31_A2W --visual-T --img-path GIIDA_A2W -i 500 --early 15
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office31 -d Office31 -s A -t D -a resnet50  --log logs/GIIDA/Office31/Office31_A2D --visual-T --img-path GIIDA_A2D -i 500 --early 15
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office31 -d Office31 -s D -t W -a resnet50  --log logs/GIIDA/Office31/Office31_D2W --visual-T --img-path GIIDA_D2W -i 500 --early 15
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office31 -d Office31 -s W -t D -a resnet50  --log logs/GIIDA/Office31/Office31_W2D --visual-T --img-path GIIDA_W2D -i 500 --early 5
wait

### OfficeHome
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Ar -t Cl -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Ar2Cl --visual-T --img-path GIIDA_Ar2Cl -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Ar -t Pr -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Ar2Pr --visual-T --img-path GIIDA_Ar2Pr -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Ar -t Rw -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Ar2Rw --visual-T --img-path GIIDA_Ar2Rw -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Cl -t Ar -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Cl2Ar --visual-T --img-path GIIDA_Cl2Ar -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Cl -t Pr -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Cl2Pr --visual-T --img-path GIIDA_Cl2Pr -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Cl -t Rw -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Cl2Rw --visual-T --img-path GIIDA_Cl2Rw -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Pr -t Ar -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Pr2Ar --visual-T --img-path GIIDA_Pr2Ar -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Pr -t Cl -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Pr2Cl --visual-T --img-path GIIDA_Pr2Cl -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Pr -t Rw -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Pr2Rw --visual-T --img-path GIIDA_Pr2Rw -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Rw -t Ar -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Rw2Ar --visual-T --img-path GIIDA_Rw2Ar -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Rw -t Cl -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Rw2Cl --visual-T --img-path GIIDA_Rw2Cl -i 1000 --early 20
wait
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py office-home -d OfficeHome -s Rw -t Pr -a resnet50 --log logs/GIIDA/OfficeHome/OfficeHome_Rw2Pr --visual-T --img-path GIIDA_Rw2Pr -i 1000 --early 20
wait

### VisDA2017
CUDA_VISIBLE_DEVICES=0 python3.6 train_GIIDA.py visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101 --per-class-eval --log logs/GIIDA/VisDA2017/ --trade-off1 1.0 --visual-T --img-path GIIDA_VisDA2017 -i 1000 --early 20 --mu 1
wait



