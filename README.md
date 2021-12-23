# **Deep GRMF**

## Introduction:
Deep learning-based graph regularized matrix factorization (**DeepGRMF**) 
integrates neural networks, graph models, and matrix-factorization techniques to predict cell response to drugs.

Evaluation of **DeepGRMF** and competing models on Genomics of Drug Sensitivity in Cancer (**GDSC**) 
and Cancer Cell Line Encyclopedia (**CCLE**) datasets show its superiority in prediction performance. 
**DeepGRMF** is also capable of predicting effectiveness of a chemotherapy regimen on patient outcomes 
for the lung cancer patients in The Cancer Genome Atlas (**TCGA**) dataset


## Usage:

No argument is used by any of the scripts. 
You will need to edit the script file to modify the inputs, outputs and hyperparameters.

Three main scripts are used for different tasks which predict drug sensitivities from:

* **New cell lines to existing drugs:** ```deepGRMF_new_cl.py```
* **New drugs to existing cell lines:** ```deepGRMF_new_drug.py```
* **New drugs to new cell lines:** ```deepGRMF_new_cl_new_drug.py```


There are two additional scripts used for cross platform training:

* **Train on GDSC  and test on CCLE:** ```deepGRMF_ccle.py``` (predict drug sensitivities of existing drugs from new cell lines)
* **Train on GDSC  and test on TCGA for survival anaylsis:** ```deepGRMF_tcga.py```

## Dataset:
All models are trained on **GDSC** dataset and then tested on **GDSC**, **CCLE** and **TCGA** dataset. 
Data preprocessing methods are described in the paper listed in **Citation** block.


## Citation:
If you find **deepGRMF** helpful, please cite the following paper: S Ren<sup>＊</sup>, Y Tao<sup>＊</sup>, K Yu, Y Xue, Schwartz<sup>†</sup>, Xinghua Lu<sup>†</sup>
[**De novo Prediction of Cell-Drug Sensitivities Using Deep Learning-based Graph Regularized Matrix Factorization**](https://www.worldscientific.com/doi/pdf/10.1142/9789811250477_0026)
Pacific Symposium on Biocomputing (***PSB***) 27:278-289(2022)
```
@inproceedings{ren2021novo,
  title={De novo Prediction of Cell-Drug Sensitivities Using Deep Learning-based Graph Regularized Matrix Factorization},
  author={Ren, Shuangxia and Tao, Yifeng and Yu, Ke and Xue, Yifan and Schwartz, Russell and Lu, Xinghua},
  booktitle={PACIFIC SYMPOSIUM ON BIOCOMPUTING 2022},
  pages={278--289},
  year={2021},
  organization={World Scientific}
}
```