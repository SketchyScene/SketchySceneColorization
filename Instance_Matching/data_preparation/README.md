# *MATCHING* Dataset Preparation

Please follow these instructions for data preparation for the *Instance Matching Module*.

1. Download the [SketchyScene](https://github.com/SketchyScene/SketchyScene) dataset following its instruction [here](https://github.com/SketchyScene/SketchyScene/tree/master/data). And you can place it anywhere you want.

1. We have provided the data in `~/Instance_Matching/data` (the three `json` files). If you want to generate them by yourself, try:

    ```
    python3 matching_data_generation.py --dataset 'all' --data_base_dir 'path/to/SketchyScene-dataset'
    ```
    - You can set `--dataset` to be *train/val/test*.
    
1. For ground-truth data **visualization**:

    ```
    python3 matching_data_visualization.py --dataset 'val' --data_base_dir 'path/to/SketchyScene-dataset'
    ```
    - Set `--image_id` to the image you want. Otherwise, a randomly selected image will be displayed.
   
