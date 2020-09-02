
Reference: [Panoptic Segmentation](https://kharshit.github.io/blog/2019/10/18/introduction-to-panoptic-segmentation-tutorial)
- Evaluate  Panoptic Quality (PQ) for MS-COCO, Cityscapes, Mapillary Vistas, ADE20k, and Indian Driving Dataset.


#### A.) Install Detectron2
- Install
    - Pytorch 1.6.0, torchvision 0.7.0, CUDA 10.1, OpenCV 4.3
    ```
       # install dependencies: (use cu101 because colab has CUDA 10.1)
       pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
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
- [Builtin Dataset](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html) Documentation
- [Custom Dataset](https://detectron2.readthedocs.io/tutorials/datasets.html) Documentation

#### B.) Setup MoDet Project

- Train
    - DETECTRON2_DATASETS variable should have [COCO panoptic dataset folder path with stuff](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html#expected-dataset-structure-for-panopticfpn) 
    ```
    export DETECTRON2_DATASETS="/media/rahul/Karmic/data"
    cd ~/workspace/tor/detectron2/projects/MoDet/

    python train_net.py --config-file configs/GRC-IEEE-Detection/faster_rcnn_R_50_FPN_1x.yaml \
                        --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025

    python train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
                        --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
    
    python train_net.py --config-file configs/COCO-PanopticSegmentation/panoptic_fpn__R_50_1x_md.yaml \ 
                        --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
    
    python train_net.py --config-file configs/Cityscapes/panoptic_fpn_R_50_1x_md.yaml 
                        --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
    ```
- Demo
    - Image, Webcam, IPCam Stream
    ```
    python run_modet.py --config-file configs/panoptic_fpn_sg_R_50_1x.yaml --input sample/input1.jpg  --opts MODEL.WEIGHTS ./output/model_0014999.pth
    
    python run_modet.py --config-file configs/panoptic_fpn_sg_R_50_1x.yaml --webcam 0                 --opts MODEL.WEIGHTS ./output/model_0014999.pth
    
    python run_modet.py --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_md.yaml --webcam "http://192.168.0.10:8080/video" \
                        --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/model_final_dbfeb4.pkl

    ```

#### C.) Evaluate Results

- Evaluate on COCO Things & Stuff Dataset for reporting

#### D.) Serve App  

##### Flask/REST Service 
- Docker App

##### Torch Serve: http://pytorch.org/serve/install.html

- **Docker** [hub for torchserve ](https://hub.docker.com/r/pytorch/torchserve/tags)

    - Sample [MaskRCNN Example](https://github.com/pytorch/serve/tree/master/examples/object_detector/maskrcnn) 
        ```
            cd sample/torchserve/ && mkdir model_store
            torch-model-archiver --model-name maskrcnn --version 1.0 --model-file maskrcnn/model.py --serialized-file maskrcnn/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth \
                                 --export-path model-store --handler object_detector --extra-files maskrcnn/index_to_name.json
        ```
        - Serve
        ``` 
            docker run --rm -it -p 3000:8080 -p 3001:8081 -v $(pwd)/model-store:/home/model-server/model-store pytorch/torchserve:0.1-cpu \
                                torchserve --start --model-store model-store --models maskrcnn=maskrcnn.mar
        ```
        - Test 
        ``` 
            curl http://127.0.0.1:3000/predictions/maskrcnn -T input1.jpg
        ```
    - Sample [Blog/Tryout](https://github.com/FrancescoSaverioZuppichini/torchserve-tryout)
        ```
            torch-model-archiver --model-name resnet34 --version 1.0 --serialized-file resnet34.pt --extra-files ./index_to_name.json,./MyHandler.py --handler my_handler.py  --export-path model-store -f
        ```    
        ```
            docker run --rm -it -p 3000:8080 -p 3001:8081 -v $(pwd)/model-store:/home/model-server/model-store pytorch/torchserve:0.1-cpu \
                        torchserve --start --model-store model-store --models resnet34=resnet34.mar
            curl -X POST http://0.0.0.0:3000/predictions/resnet34 -T input1.jpg
        ```
        - Custom Handler implementation is done above for pre/post-processing
    - Example
        ``` 
            torch-model-archiver --model-name panopticfpn --version 1.0 --model-file panopticfpn/panoptic_fpn_md.py --serialized-file panopticfpn/model_final.pth \
                                 --export-path model-store --handler object_detector --extra-files panopticfpn/index_to_name.json
        ```
        ``` 
            docker run --rm -it -p 3000:8080 -p 3001:8081 -v $(pwd)/model-store:/home/model-server/model-store pytorch/torchserve:0.1-cpu \
                                torchserve --start --model-store model-store --models panopticfpn=panopticfpn.mar
        ```
    - Stop serving
        ```  torchserve --stop ```

- **Deploy** WebApp locally using [Streamlit](https://www.streamlit.io)
    - [Realtime visualization scenario](https://github.com/streamlit/demo-self-driving)
    - [Geo information demo map](https://github.com/streamlit/demo-uber-nyc-pickups)
