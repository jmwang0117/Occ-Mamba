

<div align="center">
    <h1>🤖 OMEGA</h1>
    <h2>Efficient Occlusion-Aware Navigation for Air-Ground Robot in Dynamic Environments via State Space Model</h2>
</div>

## 📢 News

* **[05/07/2024]**: The pre-trained model can be downloaded at  [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/Edm7rZiSH3hBu_vxRJasn1wB7vPTqRWYiDgd9LcZFKbJjQ?e=mlBIfl)
* **[01/07/2024]**: OccMamba [test logs](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/ETOLWBP4jrxEvi3ZMwS8HK0Bx2yNa_xWvN-otg6ICMuzdw?e=sREpfK) and [evaluation logs](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009632_connect_hku_hk/EYGHa-8YDBlOs7nypxZEeREBStnS2eSOamYFIHU3s0sh5g?e=K4oJ5W)  on SemanticKITTI are available for download.
* **[25/06/2024]**: OccMamba code has been released.


## 📜 Introduction

**OMEGA** emerges as the pioneering navigation system tailored for AGRs in dynamic settings, with a focus on ensuring occlusion-free mapping and pathfinding. It incorporates OccMamba, a module designed to process point clouds and perpetually update local maps, thereby preemptively identifying obstacles within occluded areas. Complementing this, AGR-Planner utilizes up-to-date maps to facilitate efficient and effective route planning, seamlessly navigating through dynamic environments. 



<p align="center">
  <img src="misc/head.png" width = 60% height = 60%/>
</p>



```
@article{wangomega,
  title={OMEGA: Efficient Occlusion-Aware Navigation for Air-Ground Robot in Dynamic Environments via State Space Model},
  author={J, Wang},
  journal={arXiv preprint arXiv:2309.13882},
  year={2024}
}
```

Please kindly star ⭐️ this project if it helps you. We take great efforts to develop and maintain it 😁.

## 🛠️ Installation

## Installation
The code was tested with `python=3.6.9`, as well as `pytorch=1.10.0+cu111` and `torchvision=0.11.2+cu111`. 

Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

1. Clone the repository locally:

```
 git clone https://github.com/jmwang0117/HE-Nav.git
```
2. We recommend using **Docker** to run the project, which can reduce the burden of configuring the environment, you can find the Dockerfile in our project, and then execute the following command:
```
 docker build . -t skywalker_robot -f Dockerfile
```
3. After the compilation is complete, use our **one-click startup script** in the same directory:
```
 bash create_container.sh
```

 **Pay attention to switch docker image**

4. Next enter the container and use git clone our project
```
 docker exec -it robot bash
```
5. Then catkin_make compiles this project
```
 apt update && sudo apt-get install libarmadillo-dev ros-melodic-nlopt

```
## Run the following commands 
```
pip install pyyaml
pip install rospkg
pip install imageio
catkin_make
source devel/setup.bash
sh src/run.sh
```

You've begun this project successfully; **enjoy yourself!**


## Dataset

- [x] SemanticKITTI



## 🤗 AGR-family Works

* [OMEGA](https://jmwang0117.github.io/OMEGA/) (RA-L 2024): Prediction-boosted Planner for Aerial Reconstruction.
* [HE-Nav](https://jmwang0117.github.io/HE-Nav/) (RA-L 2024): Highly Efficient Global Planner for Aerial Coverage.
* [AGRNav](https://github.com/jmwang0117/AGRNav) (ICRA 2024): Heterogenous Multi-UAV Planner for Aerial Reconstruction.



## Acknowledgement

Many thanks to these excellent open source projects:
- [AGRNav](https://github.com/jmwang0117/AGRNav)
- [HE-Nav](https://github.com/jmwang0117/HE-Nav)
- [Prometheus](https://github.com/amov-lab/Prometheus)
- [SSC-RS](https://github.com/Jieqianyu/SSC-RS)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
- [Terrestrial-Aerial-Navigation](https://github.com/ZJU-FAST-Lab/Terrestrial-Aerial-Navigation)
- [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)

