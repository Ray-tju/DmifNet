# DmifNet
![Example 1](display/car.gif)
![Example 2](display/chair.gif)
![Example 3](display/plane.gif)
![Example 4](display/table.gif)

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
## Dataset
You can download data using
```
bash script/download_data.sh
```

## Generation
To generate meshes using a trained model, use
```
python generate.py yourpath/dmifnet.yaml
```

## Evaluation
For evaluation of the models, you can run it using

```
python eval_meshes.py yourpath/dmifnet.yaml
```
## Pretrained model
you can download our pretrained model via BaiduNetdisk

* download the [DmifNet](https://pan.baidu.com/s/1nihobjv6dW5RVt2Zw2Ycjw) and Extracted key is [3hfs]([5iwg]) 

## Quantitative Results
Method | IoU | Normal consistency | Chamfer-$L_1$ | 444
- | :-: | :-: | :-: | -:
aaa | bbb | ccc | ddd | eee| 
fff | ggg| hhh | iii | 000|

# Futher Information
Thanks for  baseline work [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks).

