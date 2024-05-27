
## [Beyond the Eye: A Relational Model for Early Dementia Detection Using Retinal OCTA Images](https://arxiv.org/abs/2311.06009) (Under Review)
<!-- [![DOI](https://img.shields.io/badge/DOI-10.1007/978--3--031--43990--2__57-darkyellow)](https://doi.org/10.1007/978-3-031-43990-2_57) -->
<!-- [![SharedIt](https://img.shields.io/badge/SharedIt-rdcu.be/dnwMc-darkyellow.svg)](https://rdcu.be/dnwMc) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2311.06009-darkyellow.svg)](https://arxiv.org/abs/2311.06009) -->

By [Shouyue Liu](https://github.com/iAaronLau), [Jinkui Hao](https://scholar.google.com/citations?user=XQqCo8QAAAAJ&hl=zh-CN), [Yonghuai Liu](https://scholar.google.com/citations?user=8J-qVlQAAAAJ&hl=zh-CN), [Huazhu Fu](https://scholar.google.com/citations?user=jCvUBYMAAAAJ&hl=zh-CN), [Xinyu Guo](https://github.com/Mr-Guowang), Shuting Zhang, and [Yitian Zhao](https://scholar.google.com/citations?user=8mULu94AAAAJ&hl=zh-CN)

![image](https://github.com/iAaronLau/PolarNet-Plus-Pytorch/blob/master/images/network.png "Flowchart")


### Contents
1. [Abstract](#Abstract)
2. [Requirements](#Requirements)
3. [Dataset](#Dataset)
4. [Citing](#Citing)


### Abstract

Early detection of dementia, such as Alzheimer's disease (AD) or mild cognitive impairment (MCI), is essential to enable timely intervention and potential treatment. Accurate detection of AD/MCI is challenging due to the high complexity, cost, and often invasive nature of current diagnostic techniques, which limit their suitability for large-scale population screening. Given the shared embryological origins and physiological characteristics of the retina and brain, retinal imaging is emerging as a potentially rapid and cost-effective alternative for the identification of individuals with or at high risk of AD. In this paper, we present a novel PolarNet+ that uses retinal optical coherence tomography angiography (OCTA) to discriminate early-onset AD (EOAD) and MCI subjects from controls. Our method first maps OCTA images from Cartesian coordinates to polar coordinates, allowing approximate sub-region calculation to implement the clinician-friendly early treatment of diabetic retinopathy study (ETDRS) grid analysis. We then introduce a multi-view module to serialize and analyze the images along three dimensions for comprehensive, clinically useful information extraction. Finally, we abstract the sequence embedding into a graph, transforming the detection task into a general graph classification problem. A regional relationship module is applied after the multi-view module to excavate the relationship between the sub-regions. Such regional relationship analyses validate known eye-brain links and reveal new discriminative patterns. The proposed model is trained, tested, and validated on four retinal OCTA datasets, including 1,671 participants with AD, MCI, and healthy controls. Experimental results demonstrate the performance of our model in detecting AD and MCI with an AUC of 88.69% and 88.02%, respectively. Our results provide evidence that retinal OCTA imaging, coupled with artificial intelligence, may serve as a rapid and non-invasive approach for large-scale screening of AD and MCI.


### Requirements

1. System Requirements:
	- NVIDIA GPUs, CUDA supported.
	- Ubuntu 20.04 workstation or server
	- Anaconda environment
	- Python 3.9
	- PyTorch 2.0 
	- Git

2. Installation:
   - `git clone https://github.com/iAaronLau/PolarNet-Plus-Pytorch.git`
   - `cd ./PolarNet-Plus-Pytorch`
   - `conda env create -f environment.yaml`


### Dataset

Please put the root directory of your dataset into the folder ./data. The root directory contain the two subfolder now: AD and control. The most convenient way is to follow the sample file structure, as follows:

```
|-- data
    |-- root directory
        |-- AD
        |-- control
            |-- ID_name
                |-- macular3_3 or 3x3
                    |-- *SVC.png
                    |-- *DVC.png
                    |-- *choriocapillaris.png or *CC.png
                    |-- ... 
```

You can also change the file structure. Note that you need to change the data processing function to ensure the data can be obtained correctly. 

Due to the method need the multiple inputs, i.e., SVC, DVC and choriocapillaris, so the most important thing is that you need specify the filter words for file name of SVC, DVC, and choriocapillaris. Please make sure the three filter words are in the right order.

### Citing 

If you find our paper useful in your research, please consider citing:

```
@inproceedings{liu2023polar,
  title={Polar-Net: A Clinical-Friendly Model for Alzheimer’s Disease Detection in OCTA Images},
  author={Liu, Shouyue and Hao, Jinkui and Xu, Yanwu and Fu, Huazhu and Guo, Xinyu and Liu, Jiang and Zheng, Yalin and Liu, Yonghuai and Zhang, Jiong and Zhao, Yitian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={607--617},
  year={2023},
  organization={Springer}
}
```

### License
MIT License
