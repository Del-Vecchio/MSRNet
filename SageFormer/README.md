# SageFormer: Series-Aware Framework for Long-Term Multivariate Time-Series Forecasting

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fzhangzw16%2FSageFormer&label=VISITORS&labelColor=%232ccce4&countColor=%23697689)

This repository contains the code for the paper "[SageFormer: Series-Aware Framework for Long-Term Multivariate Time-Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10423755)" by Zhenwei Zhang, Linghang Meng, and Yuantao Gu, published in the IEEE Internet of Things Journal.

## Introduction

SageFormer is a novel series-aware graph-enhanced Transformer model designed for long-term forecasting of multivariate time-series (MTS) data. With the proliferation of IoT devices, MTS data has become ubiquitous, necessitating advanced models to forecast future behaviors. SageFormer addresses the challenge of capturing both intra- and inter-series dependencies, enhancing the predictive performance of Transformer-based models.
<div align=center>
<img width="1145" alt="Screenshot 2024-02-20 at 14 56 56" src="https://github.com/zhangzw16/SageFormer/assets/26004183/941c5e6d-d261-41fb-bf20-4211c4fa6d9e">
<img width="896" alt="Screenshot 2024-02-20 at 14 58 19" src="https://github.com/zhangzw16/SageFormer/assets/26004183/3ed21ee8-e11f-4da9-ad6d-c80413b33b07">
</div>

## Usage
To train and evaluate the SageFormer model:

- Clone this repository
- Download datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) and place them in the `./dataset` folder
- Create a virtual environment and activate it
- Install requirements `pip install -r requirements.txt`
- Run scripts in the `./scripts` folder to train and evaluate the model, for example:
    ```bash
    sh scripts/long_term_forecast/ECL_script/SageFormer.sh
    ``` 
- Model checkpoints and logs will be saved to outputs folder

## Contacts
For any questions, please contact the authors at `zzw20 [at] mails.tsinghua.edu.cn` or write a [discussion on github](https://github.com/zhangzw16/SageFormer/discussions).

## Citation
If you find this code or paper useful for your research, please cite:
```bibtex
@ARTICLE{zhang2024sageformer,
  author={Zhang, Zhenwei and Meng, Linghang and Gu, Yuantao},
  journal={IEEE Internet of Things Journal}, 
  title={SageFormer: Series-Aware Framework for Long-Term Multivariate Time Series Forecasting}, 
  year={2024},
  doi={10.1109/JIOT.2024.3363451}}
```

# Acknowledgement

This library is constructed based on the following repos:
- https://github.com/thuml/Time-Series-Library
- https://github.com/PatchTST/PatchTST
