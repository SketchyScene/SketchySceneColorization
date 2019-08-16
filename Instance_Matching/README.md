# Instance Matching Module

This folder holds the code and dataset for *Instance Matching Module* in the SketchyScene Colorization system.

## Requirements
- Python 3
- Tensorflow (>= 1.3.0)
- scipy
- PIL


## Preparations

- Please follow the instructions [here](/Instance_Matching/data_preparation) for *MATCHING* dataset preparation.
- If you want to train the network, please follow the the instructions [here](/Instance_Matching/models) for pre-trained model preparation.


## Training

After the preparations, run:
```
python3 matching_main.py --mode 'train' --data_base_dir 'path/to/SketchyScene-dataset'
```

  - Set `--model` to *'deeplab_v3plus'*, *'fcn_8s'* or *'segnet'* for other alternatives. The default is *deeplab* (Deeplab-v2 model).
  
  
## Evaluation

Please follow these steps for the evaluation:

1. Please make sure the trained model has been placed under `outputs/snapshots`. We have also provided our trained model, which can be downloaded [here](https://drive.google.com/drive/folders/1wGU3vln9Nc_Z2NV2F5nyt_2NbqDsvuRO?usp=sharing).

1. Our *Instance Matching Module* consists of *Mask R-CNN* for instance segmentation. Since the evaluation will performed on a sketch image for multiple times (several testing sentences), it is recommended to generate the segmented data first and reuse it to avoid the redundant time caused by the repeated forwarding procedure. To do so, please go to the [SketchyScene](https://github.com/SketchyScene/SketchyScene) repository for some preparations:

    1. Strictly follow the instructions in the [Instance-Segmentation](https://github.com/SketchyScene/SketchyScene#instance-segmentation) part to setup the *Mask R-CNN* model. 

    1. After setuping the *Instance Segmentation* model, place the two json files `data/sentence_instance_val.json` and `data/sentence_instance_test.json` of *MATCHING* dataset to `SketchyScene/Instance_Segmentation/data/`, and then run the following command under `SketchyScene/Instance_Segmentation` to generate the segmented data:

        ```
        python3 segment_data_generation.py --data_basedir 'path/to/SketchyScene-dataset'
        ```
        - Set `--use_edgelist` to 0 if you don't want to use *edgelist*. The default is 1 and you must strictly follow the [instructions](https://github.com/SketchyScene/SketchyScene/tree/master/Instance_Segmentation/libs/edgelist_utils_matlab) to prepare for the edgelist data.
        
    1. Finally you will find the segmented data under `outputs/inst_segm_output_data`. Then place them to `SketchySceneColorization/Instance_Matching/outputs/inst_segm_output_data/test(val)/seg_data`.
    
1. After these, run:

    ```
    python3 matching_main.py --mode 'eval' --dataset 'val' --data_base_dir 'path/to/SketchyScene-dataset'
    ```

    - Set `--dataset` to *val/test*.
    - Set `--visualized` to 1 if you want to see visual results during evaluation. The default is 0.


## Inference

Make sure the trained model and the segmented data have been placed correctly as in the [Evaluation](#evaluation) part. Then run:

```
python3 matching_main.py --mode 'inference' --data_base_dir 'path/to/SketchyScene-dataset' \
                         --dataset 'val' --image_id 2 --instruction 'your instruction'
```

- Set `--dataset` to *val/test*.
- Set `--image_id` to the image you want.
- Input your instruction after `--instruction`.

**For example**, we have provided a sketch image and its segmented data under `examples` directory. You can try it:

```
python3 matching_main.py --mode 'inference' --data_base_dir 'examples' --seg_data_dir 'examples' \
                         --dataset 'test' --image_id 30 \
                         --instruction 'all the grasses are dark green'
```

- Try other instructions for multiple times by changing the text after `--instruction` :)

If everything goes well, you can obtain the result like this:
![example](/Instance_Matching/figures/result_example.png)

## Citation

Please cite the corresponding paper if you found the datasets or code useful:

```
TBA......
```
