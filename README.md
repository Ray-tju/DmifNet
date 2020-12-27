# DmifNet
![Example 1](display/car.gif)
![Example 2](display/chair.gif)
![Example 3](display/plane.gif)
![Example 4](display/table.gif)

## Citing this work
```
@inproceedings{DmifNet,
          title={{DmifNet: 3D Shape Reconstruction based onDynamic Multi–Branch Information Fusion}},
          author={Lei Li and Suping Wu},
          booktitle={Proceedings IEEE Conf. on International Conference on Pattern Recognition (ICPR)},
          year={2020}
        }
```
## Installation
First you have to make sure that you have all dependencies in place.

You can create an anaconda environment called `dmifnet_space` using
```
conda env create -f dmif_env.yaml
conda activate dmifnet_space
```

Then, compile the extension modules.
```
python set_env_up.py build_ext --inplace
```

## Generation
To generate meshes using a trained model, use
```
python generate.py yourpath/dmifnet.yaml
```

## Training
```
python train.py yourpath/dmifnet.yaml
```

## DataSet
There is no space in the cloud disk to upload our dataset, you can contact me by email to get the dataset, or you can check the baseline work Onet to download the dataset.

## Evaluation
For evaluation of the models, you can run it using

```
python eval_meshes.py yourpath/dmifnet.yaml
```
## Pretrained model
you can download our pretrained model via BaiduNetdisk

* download the [DmifNet](https://pan.baidu.com/s/1nihobjv6dW5RVt2Zw2Ycjw) and Extracted key is [3hfs]([5iwg]) 

## Quantitative Results
Method | Intersection over Union | Normal consistency | Chamfer distance 
:-: | :-: | :-: | :-: 
3D-R2N2 | 0.493 | 0.695 | 0.278  
Pix2Mesh | 0.480 | 0.772 | 0.216 
AtlasNet | -- | 0.811 | 0.175 
ONet | 0.571 | 0.834 | 0.215
DmifNet | 0.607 | 0.846 | 0.185

# Futher Information
If you have any problems with the code, please list the problems you encountered in the issue area, and I will reply you soon.
Thanks for  baseline work [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks).

