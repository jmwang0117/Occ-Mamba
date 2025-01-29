<div align="center">
    <h1>🤖 OMEGA</h1>
    <h2>Efficient Occlusion-Aware Navigation for Air-Ground Robot in Dynamic Environments via State Space Model</h2> <br>
    <a href='https://arxiv.org/abs/2408.10618'><img src='https://img.shields.io/badge/arXiv-OMEGA-green' alt='arxiv'></a>
     <a href='https://jmwang0117.github.io/OMEGA/'><img src='https://img.shields.io/badge/Project_Page-OMEGA-green' alt='Project Page'></a>
    
</div>

## 🤗 AGR-Family Works

* [OccRWKV](https://jmwang0117.github.io/OccRWKV/) (ICRA 2025.01): The First RWKV-based 3D Semantic Occupancy Network
* [OMEGA](https://jmwang0117.github.io/OMEGA/) (RA-L 2024.12): The First AGR-Tailored Dynamic Navigation System.
* [HE-Nav](https://jmwang0117.github.io/HE-Nav/) (RA-L 2024.09): The First AGR-Tailored ESDF-Free Navigation System.
* [AGRNav](https://github.com/jmwang0117/AGRNav) (ICRA 2024.01): The First AGR-Tailored Occlusion-Aware Navigation System.


## 🎉 Chinese Media Reports/Interpretations
* [AMOV Lab Research Scholarship](https://mp.weixin.qq.com/s/AXbW3LDgsl9knQBIMwIpvA) -- 2024.11: 5000 RMB
* [AMOV Lab Research Scholarship](https://mp.weixin.qq.com/s/PUwY04sMpVmz30kSn6XdzQ) -- 2024.10: 5000 RMB

## 📢 News
* **[03/07/2024]**: OMEGA's simulation logs are available for download:

<div align="center">

| Simulation Results | Experiment Log |
|:------------------------------------------------------------------:|:----------:|
|OMEGA | [link](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/u3009632_connect_hku_hk/EYQCfCo-UdJEt6p7H6eyhioBgJd2rIWKbb1IEqVo2hEjkg?e=zxFxMB) |
|AGRNav|  [link](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/u3009632_connect_hku_hk/EYu6dtz6G6NHjjSvAUpuTjUBZC-Rmp2CjosK3qeVtijhPQ?e=P1YrvX) |
|TABV| [link](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/u3009632_connect_hku_hk/EVLIa9V_gLlMs6vOfvzhE28Bp58W6u_KH5p353SoAdxjjA?e=k3GxRP) |

</div>

* **[01/07/2024]**: OccMamba's test and evaluation logs are available for download:

<div align="center">

| OccMamba Results | Experiment Log |
|:------------------------------------------------------------------:|:----------:|
|OccMamba on the SemanticKITTI hidden official test dataset | [link](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/u3009632_connect_hku_hk/EReNqjk3AehAuvCllef7I6ABEbyl1yu2oPuQ2eYcv5Ad5A?e=RQMFSt) |
| OccMamba test log|  [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/ETOLWBP4jrxEvi3ZMwS8HK0Bx2yNa_xWvN-otg6ICMuzdw?e=sREpfK) |
|OccMamba evaluation log | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/EYGHa-8YDBlOs7nypxZEeREBStnS2eSOamYFIHU3s0sh5g?e=K4oJ5W) |

</div>

* **[28/06/2024]**: The pre-trained model can be downloaded at  [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/Edm7rZiSH3hBu_vxRJasn1wB7vPTqRWYiDgd9LcZFKbJjQ?e=mlBIfl)
* **[25/06/2024]**: We have released the code for OccMamba, a key component of OMEGA!


## 📜 Introduction

**OMEGA** emerges as the pioneering navigation system tailored for AGRs in dynamic settings, with a focus on ensuring occlusion-free mapping and pathfinding. It incorporates OccMamba, a module designed to process point clouds and perpetually update local maps, thereby preemptively identifying obstacles within occluded areas. Complementing this, AGR-Planner utilizes up-to-date maps to facilitate efficient and effective route planning, seamlessly navigating through dynamic environments. 



<p align="center">
  <img src="misc/head.png" width = 60% height = 60%/>
</p>

<br>

```
@article{wang2025omega,
  author={Wang, Junming and Guan, Xiuxian and Sun, Zekai and Shen, Tianxiang and Huang, Dong and Liu, Fangming and Cui, Heming},
  journal={IEEE Robotics and Automation Letters}, 
  title={OMEGA: Efficient Occlusion-Aware Navigation for Air-Ground Robots in Dynamic Environments via State Space Model}, 
  year={2025},
  volume={10},
  number={2},
  pages={1066-1073},
  publisher={IEEE}
}
```
<br>

Please kindly star ⭐️ this project if it helps you. We take great efforts to develop and maintain it 😁.

## 🔧 Hardware List

<div align="center">

| Hardware | Link |
|:------------------:|:----------:|
| AMOV Lab P600 UAV | [link](https://www.amovlab.com/product/detail?pid=76) |
| AMOV Lab Allapark1-Jetson Xavier NX | [link](https://www.amovlab.com/product/detail?pid=77) |
| Wheeltec R550 ROS Car | [link](https://lubancat.wheeltec.net/zh-cn/main/neirong/01%E6%9C%BA%E5%99%A8%E4%BA%BA%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D/01%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D.html) |
| Intel RealSense D435i | [link](https://www.intelrealsense.com/depth-camera-d435i/) |
| Intel RealSense T265 | [link](https://www.intelrealsense.com/visual-inertial-tracking-case-study/) |
| TFmini Plus | [link](https://en.benewake.com/TFminiPlus/index_proid_323.html) |

</div>

❗ Considering that visual positioning is prone to drift in the Z-axis direction, we added TFmini Plus for height measurement. Additionally, **GNSS-RTK positioning** is recommended for better localization accuracy.

🤑 Our customized Aerial-Ground Robot cost about **RMB 70,000**.

## 🛠️ Installation

```
conda create -n occmamba python=3.10 -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install spconv-cu120
pip install tensorboardX
pip install dropblock
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -U openmim
mim install mmcv-full
pip install mmcls==0.25.0
```
> [!NOTE]
> Please refer to [Vision-Mamba](https://github.com/hustvl/Vim) for more installation information.

## 💽 Dataset

- [x] SemanticKITTI

Please download the Semantic Scene Completion dataset (v1.1) from the [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html) and extract it.

Or you can use [voxelizer](https://github.com/jbehley/voxelizer) to generate ground truths of semantic scene completion.

The dataset folder should be organized as follows.
```angular2
SemanticKITTI
├── dataset
│   ├── sequences
│   │  ├── 00
│   │  │  ├── labels
│   │  │  ├── velodyne
│   │  │  ├── voxels
│   │  │  ├── [OTHER FILES OR FOLDERS]
│   │  ├── 01
│   │  ├── ... ...
```

## 🤗 Getting Start
Clone the repository:
```
git clone https://github.com/jmwang0117/Occ-Mamba.git
```


### Train OccMamba Net

```
$ cd <root dir of this repo>
$ bash run_train.sh
```
### Validation


```
$ cd <root dir of this repo>
$ bash run_val.sh
```
### Test

Since SemantiKITTI contains a hidden test set, we provide test routine to save predicted output in same format of SemantiKITTI, which can be compressed and uploaded to the [SemanticKITTI Semantic Scene Completion Benchmark](http://www.semantic-kitti.org/tasks.html#ssc). You can provide which checkpoints you want to use for testing. We used the ones that performed best on the validation set during training. For testing, you can use the following command.

```
$ cd <root dir of this repo>
$ bash run_test.sh
```


## 🏆 Acknowledgement
Many thanks to these excellent open source projects:
- [Vision-Mamba](https://github.com/hustvl/Vim)
- [AGRNav](https://github.com/jmwang0117/AGRNav)
- [Prometheus](https://github.com/amov-lab/Prometheus)
- [SSC-RS](https://github.com/Jieqianyu/SSC-RS)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
- [Terrestrial-Aerial-Navigation](https://github.com/ZJU-FAST-Lab/Terrestrial-Aerial-Navigation)
- [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)

