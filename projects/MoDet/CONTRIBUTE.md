### Panoptic Segmentation 

#### [Reference](https://github.com/Angzz/awesome-panoptic-segmentation)

##### 1. Definition
A unified general view of semantic image segmentation. PS algorithm heuristically combines the semantic segmentation of *Stuff* with instance segmentation of *Thing* in the scene
- Semantic segmentation: pixelwise segmentation of stuff classes using a fully conv net with dilations
- Instance segmentation: A region proposal based object detection task

<div align="center"><img src="./sample/readme/panoptic_segmentation_overview2.png" width="900" height="180"></div>

##### 2. Structure Overview
<div align="center"><img src="./sample/readme/panoptic_structure.png" width="800" height="240"></div>

from [UPSNet](https://arxiv.org/pdf/1901.03784.pdf).

##### 3. Task
Each pixel is assigned a semantic label and an instance id. 
- Pixel with same label and id belong to the same object for stuff labels, the instance id is ignored
- Unlike instance segmentation, object segmentation must be non-overlapping
- it is not a multi-task problem but rather a "unified view" or "strict generalization" of semantic image segmentation 

##### 4. Dataset
datasets which contains both semantic and instance annotations

* [COCO-Panoptic](http://cocodataset.org/)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [Mapillary Vistas](https://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html)
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [IDD20K](http://idd.insaan.iiit.ac.in/)


##### 5. Metrics: [Panoptic Quality](https://cocodataset.org/#panoptic-eval)
* ``PQ`` are the standard metrics described in [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868.pdf). 
`` PQ = Seg Quality (SQ) x Recognitions Quality (RQ) ``
<div align="center" width="10" height="5"><img src="./sample/readme/pq_metric.png" width="600" height="150"></div>

* ``PC`` are the standard metrics described in [DeeperLab](https://arxiv.org/pdf/1902.05093).
<div align="center" width="10" height="5"><img src="./sample/readme/pc_metric.png" width="600" height="207"></div>

##### 6. Competition Leaderboard 
* [COCO 2018 Panoptic Segmentation Task (ECCV 2018 Workshop, Closed)](http://cocodataset.org/index.htm#panoptic-2018)


##### 7. Papers 

* (Arxiv 2020) **EfficientPS:** Rohit Mohan and	Abhinav Valada, "EfficientPS: Efficient Panoptic Segmentation", 
arXiv preprint arXiv:2004.02307, 2020. [[Paper/Code](http://panoptic.cs.uni-freiburg.de/)]

* (CVPR 2020) **Video Panoptic Segmentation:** Dahun Kim, Sanghyun Woo, Joon-Young Lee, In So Kweon <br />"VPSNet for Video Panoptic Segmentation." CVPR (2020). [[Paper/Code](https://github.com/mcahny/vps)]

* (AAAI 2020) **SOGNet:** Yibo Yang, Hongyang Li, Xia Li, Qijie Zhao, Jianlong Wu, Zhouchen Lin.<br />"SOGNet: Scene Overlap Graph Network for Panoptic Segmentation." AAAI (2020). [[paper](https://arxiv.org/pdf/1911.07527.pdf)]

* (CVPR 2019) **Panoptic Segmentation:** Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, Piotr Dollár.<br />"Panoptic Segmentation." CVPR (2019). [[paper](https://arxiv.org/pdf/1801.00868.pdf)]

---

#### A. Working with [PyTorch](https://pytorch.org/get-started/locally/) and [CUDA 10.1](https://www.tensorflow.org/install/gpu#install_cuda_with_apt)
- Pytorch 1.6.0, torchvision 0.7.0, CUDA 10.1, OpenCV 4.3
```
   # install dependencies: (use cu101 because colab has CUDA 10.1)
   pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

#### B. Python [Virtual Environment Wrapper](https://medium.com/the-andela-way/configuring-python-environment-with-virtualenvwrapper-8745c2895745)
- Setup virtualenvwrapper
    ``` 
    mkvirtualenv det2 -p python3 
    pip install -r requirements
    ```
- [Install Detectron2](https://detectron2.readthedocs.io/tutorials/install.html)
    ```
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
    pip install -r requirements
    ```
    OR
    ```
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2 && python setup.py build develop
    ```

#### C. Working with the Upstream GIT code [reference](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository)

1. Add the "upstream" to your cloned repository ("origin"):
 ```git remote add upstream git@github.com:facebookresearch/detectron2.git```

2. Fetch the commits (and branches) from the "upstream":
 ```git fetch upstream ```

3. List and Switch to the "master" branch of your fork ("origin"):
 ```git branch -a```
 ```git checkout master ```

4. Stash the changes of your "master" branch:
 ```git stash ```

5. Merge the changes from the "master" branch of the "upstream" into your the "master" branch of your "origin":
 ```git merge upstream/master ```

6. Resolve merge conflicts if any and commit your merge
 ```git commit -am "Merged from upstream" ```

7. Push the changes to your fork
 ```git push ```

8. Get back your stashed changes (if any)
 ```git stash pop ```

9. You're done! Congratulations!

GitHub also provides instructions for this 