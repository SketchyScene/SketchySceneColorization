# SketchySceneColorization - SIGA2019

[Paper](http://mo-haoran.com/files/SIGA19/SketchColorization_paper_SA2019.pdf) | [Supplementary Material](http://mo-haoran.com/files/SIGA19/SketchColorization_supplementary_SA2019.pdf) | [Project Page](https://sketchyscene.github.io/SketchySceneColorization/)

This repository hosts the datasets and the code for the SketchyScene Colorization system (**SIGGRAPH Asia 2019**). Please refer to our paper for more information: [Language-based Colorization of Scene Sketches](http://sweb.cityu.edu.hk/hongbofu/doc/language-based_sketch_colorization_SA19.pdf).

![examples](/figures/teaser.gif)

### System Overview

Our system supports two-mode interactive colorization for a given input scene sketch and text-based colorization instructions, using three models, namely, the instance matching model, foreground colorization model, and background colorization model.

![system](/figures/system_overview.png)


## Outline
- [Requirements](#requirements)
- [Preparations](#preparations)
    - [Instance Matching](#instance-matching)
    - [Foreground Instance Colorization](#foreground-instance-colorization)
    - [Background Colorization](#background-colorization)
- [Colorizing With The Whole Pipeline](#colorizing-with-the-whole-pipeline)
- [Citation](#citation)

## Requirements
- Python 3
- Tensorflow (>= 1.3.0)
- scipy
- PIL
- skimage

## Preparations

Please follow the instructions in the following three sections (Instance Matching, Foreground Instance Colorization, and Background Colorization) to download the dataset and pre-trained models and place them in the correct directories.


### Instance Matching

For the details of *MATCHING* dataset and the code, please refer to the [Instance_Matching](/Instance_Matching) directory.

![matching](/figures/instance_match_network.png)

### Foreground Instance Colorization

For the details of *FOREGROUND* dataset and the code, please refer to the [Foreground_Instance_Colorization](/Foreground_Instance_Colorization) directory.

![foreground](/figures/inst_color_network.png)

### Background Colorization

For the details of *BACKGROUND* dataset and the code, please refer to the [Background_Colorization](/Background_Colorization) directory.

![background](/figures/bg_color_network.png)

## Colorizing With The Whole Pipeline

Our system allows users to colorize the sketches through language instructions. If the result is not satisfactory, users can also withdraw the last instruction.

:fire: **We have provided some test examples** in `examples` directory.

1. To *colorize* a sketch, run the command like:

    ```
    python3 sketchyscene_colorization_main.py --image_id 9996 \
                                              --instruction 'the bus is orange with gray windows'
    ```
    - Set `image_id` to the sketch you want.
    - Try other instructions by changing the `instruction`.
    
    You will see the results in `outputs` directory.
  
2. To *withdraw* the last instruction, run the command like:

    ```
    python3 sketchyscene_colorization_main.py --command 'withdraw' --image_id 9996
    ```
    
    See what happens in `outputs` directory :)

![examples](/figures/result2.gif)


## Citation

Please cite the corresponding paper if you found the datasets or code useful:

```
@article{zouSA2019sketchcolorization,
  title = {Language-based Colorization of Scene Sketches},
  author = {Zou, Changqing and Mo, Haoran and Gao, Chengying and Du, Ruofei and Fu, Hongbo},
  journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH Asia 2019)},
  year = {2019},
  volume = 38,
  number = 6,
  pages = {233:1--233:16}
}

@inproceedings{zou2018sketchyscene,
  title={Sketchyscene: Richly-annotated scene sketches},
  author={Zou, Changqing and Yu, Qian and Du, Ruofei and Mo, Haoran and Song, Yi-Zhe and Xiang, Tao and Gao, Chengying and Chen, Baoquan and Zhang, Hao},
  booktitle={Proceedings of the european conference on computer vision (ECCV)},
  pages={421--436},
  year={2018}
}
```
