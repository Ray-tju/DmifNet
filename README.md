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
\begin{table*}[t]
\begin{center}
\caption{ \textbf{Single Image 3D Reconstruction Results on ShapeNet}. Quantitative evaluations on the ShapeNet under IoU, Normal consistency and Chamfer distance. We observe that our method approach outperforms other state-of-the-art learning based methods in Normal consistency and IoU.}\label{tab:cap}. \label{tab:cap}
\setlength{\tabcolsep}{0.80mm}{
\begin{tabular}{ccccccccccccccccc} 
  \hline
  \textbf{IoU $\uparrow$} & Airplane & Bench & Cabinet & Car & Chair & Display & Lamp & Loudspeaker & Rifle & Sofa & Table & Telephone & Vessel  & Mean
  \\
  \hline
  3D-R2N2 \cite{b2} ECCV'16       & 0.426 & 0.373 & 0.667 & 0.661 & 0.439 & 0.440 & 0.281 & 0.611 & 0.375 & 0.626 &0.420 &0.6118 &0.482  & 0.493  \\
  Pix2Mesh  \cite{b10} ECCV'18  & 0.420 & 0.323 & 0.664 & 0.552 & 0.396 & 0.490 & 0.323 & 0.599 & 0.402& 0.613 & 0.395 & 0.661 &0.397 & 0.480 \\
   AtlasNet  \cite{b16} CVPR'18  & - & - & - & - & - & - & -& - & - & - & - & - &- &-  \\
  ONet  \cite{b17} CVPR'19  & 0.571 & 0.485 & 0.733 & 0.737 & 0.501 & 0.471 & 0.371& 0.647 & 0.474 & 0.680 & 0.506 & 0.720 &0.530 &0.571  \\
 \textbf{Our}   & \textbf{0.603} & \textbf{0.512}& \textbf{0.753}&\textbf{0.758}& \textbf{0.542}& \textbf{0.560} & \textbf{0.416}& \textbf{0.675} & \textbf{0.493} & \textbf{0.701}& \textbf{0.550} & \textbf{0.750}& \textbf{0.574} &\textbf{0.607}  \\
  \hline
  \textbf{Normal Consistency $\uparrow$} & Airplane & Bench & Cabinet & Car & Chair & Display & Lamp & Loudspeaker & Rifle & Sofa & Table & Telephone & Vessel  & Mean\\
  \hline
   3D-R2N2 \cite{b2} ECCV'16       & 0.629 &0.678& 0.782 & 0.714 & 0.663 & 0.720 & 0.560 & 0.711 & 0.670 &0.731&0.732 &0.817 &0.629  & 0.695  \\
  Pix2Mesh  \cite{b10} ECCV'18  & 0.759 & 0.732 & 0.834 & 0.756 & 0.746 & 0.830 & 0.666 & 0.782 & 0.718& 0.820 & 0.784 & 0.907 &0.699& 0.772 \\
   AtlasNet  \cite{b16} CVPR'18  & 0.836 & 0.779 & 0.850 & 0.836 & 0.791 & 0.858 & 0.694& 0.825 & 0.725 & 0.840 & 0.832 & 0.923 &0.756 &0.811  \\
  ONet  \cite{b17} CVPR'19  & 0.840 & 0.813 & 0.879 & 0.852 & 0.823 & 0.854 & 0.731& 0.832 & 0.766 & 0.863 & 0.858 & 0.935 &0.794 &0.834  \\
 \textbf{Our}   & \textbf{0.853} & \textbf{0.821}& \textbf{0.885}&\textbf{0.857}& \textbf{0.835}& \textbf{0.872} & \textbf{0.758}& \textbf{0.847} & \textbf{0.781} & \textbf{0.873}& \textbf{0.868} & \textbf{0.936}& \textbf{0.808} &\textbf{0.846}  \\
  \hline
  \textbf{Chamfer-$L_1$ $\downarrow$} & Airplane & Bench & Cabinet & Car & Chair & Display & Lamp & Loudspeaker & Rifle & Sofa & Table & Telephone & Vessel  & Mean\\
  \hline
   3D-R2N2 \cite{b2} ECCV'16       & 0.227 &0.194& 0.217 & 0.213 & 0.270 & 0.314 & 0.778 & 0.318 & 0.183 &0.229&0.239 &0.195 &0.238  & 0.278  \\
  Pix2Mesh  \cite{b10} ECCV'18  & 0.187 & 0.201 & 0.196 & 0.180 & 0.265 & 0.239 & 0.308 & 0.285 & 0.164& 0.212 & 0.218 & 0.149 &0.212& 0.216 \\
   AtlasNet  \cite{b16} CVPR'18   & \textbf{0.104} & \textbf{0.138}& 0.175&\textbf{0.141}& 0.209& \textbf{0.198} & \textbf{0.305}& \textbf{0.245} & \textbf{0.115} & \textbf{0.177}& 0.190 & 0.128& \textbf{0.151} &\textbf{0.175}  \\
  ONet  \cite{b17} CVPR'19  & 0.147 & 0.155 & 0.167 & 0.159 & 0.228 & 0.278 & 0.479& 0.300 & 0.141 & 0.194 & 0.189 & 0.140 &0.218 &0.215  \\
 \textbf{Our}   & 0.131 & 0.141 &\textbf{0.149} & 0.142 &\textbf{0.203} & 0.220 & 0.351& 0.263 & 0.135 & 0.181 & \textbf{0.173}  &\textbf{0.124} &0.189 &0.185  \\
  \hline
  \multicolumn{4}{l}{$^{\mathrm{a}}$ The Bold-faced numbers represent the best results.}
\end{tabular}}
\end{center}
\end{table*}

# Futher Information
Thanks for  baseline work [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks).

