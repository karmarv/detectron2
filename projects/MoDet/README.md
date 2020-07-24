
Reference: [Panoptic Segmentation](https://kharshit.github.io/blog/2019/10/18/introduction-to-panoptic-segmentation-tutorial)
- Evaluate  Panoptic Quality (PQ) for MS-COCO, Cityscapes, Mapillary Vistas, ADE20k, and Indian Driving Dataset.


#### A.) Install Detectron2


#### B.) Setup MoDet Project

- Train
    - DETECTRON2_DATASETS variable should have [COCO panoptic dataset folder path with stuff](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html#expected-dataset-structure-for-panopticfpn) 
    ```
    export DETECTRON2_DATASETS="/media/rahul/Karmic/data"
    
    ./train_net.py --config-file configs/panoptic_fpn_sg_R_50_1x.yaml  \
    --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
    ```
- Demo
    ```
    python run_modet.py --config-file configs/panoptic_fpn_sg_R_50_1x.yaml --input sample/input1.jpg  --opts MODEL.WEIGHTS ./output/model_0014999.pth
    python run_modet.py --config-file configs/panoptic_fpn_sg_R_50_1x.yaml --webcam 0                 --opts MODEL.WEIGHTS ./output/model_0014999.pth
    python run_modet.py --config-file configs/panoptic_fpn_sg_R_50_1x.yaml --webcam "http://192.168.0.29:8080/video" --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/model_final_dbfeb4.pkl

    ```