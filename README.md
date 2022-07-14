# Continuous Facial Motion Deblurring

This repository is the Official Pytorch Implementation of [Continuous Facial Motion Deblurring].
```
T.B. Lee, S. Han and Y.S. Heo, "Continuous Facial Motion Deblurring," in IEEE Access, 2022.
```

![CFMDGAN](/images/overview_cfmd_gan.png)

## Test results on 300VW dataset
| Blurry Image  | Ground-Truth  | Ours 7 Frames | Ours 51 Frames |
| ------------- | ------------- | ------------- | ------------- |
| ![](/images/009_blur007_000044/input.png)  | ![](/images/009_blur007_000044/gt.gif) | ![](/images/009_blur007_000044/cfmd_07.gif) | ![](/images/009_blur007_000044/cfmd_51.gif) |
| ![](/images/039_blur007_000039/input.png)  | ![](/images/039_blur007_000039/gt.gif) | ![](/images/039_blur007_000039/cfmd_07.gif) | ![](/images/039_blur007_000039/cfmd_51.gif) |
| ![](/images/158_blur007_002879/input.png)  | ![](/images/158_blur007_002879/gt.gif) | ![](/images/158_blur007_002879/cfmd_07.gif) | ![](/images/158_blur007_002879/cfmd_51.gif) |
| ![](/images/522_blur007_000019/input.png)  | ![](/images/522_blur007_000019/gt.gif) | ![](/images/522_blur007_000019/cfmd_07.gif) | ![](/images/522_blur007_000019/cfmd_51.gif) |

## Test results on REDS, LAI examples
| Blurry Image  | Ours 51 Frames |
| ------------- | ------------ |
| ![](/images/Lai_11/input.png)  | ![](/images/Lai_11/cfmd_51.gif) | 
| ![](/images/REDS_01/input.png)  | ![](/images/REDS_01/cfmd_51.gif) | 

## 1. Dependencies
+ Python >= 3.7
+ Pytorch >= 1.12.0


## 2. Training
### 1) Preapre training data
