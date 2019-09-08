# Background Colorization Module

This directory hosts the code and dataset for *Background Colorization Module* in the SketchyScene Colorization system.

## Requirements
- Python 3
- Tensorflow (>= 1.3.0)
- PIL


## Preparations

- Please follow the instructions [here](/Background_Colorization/data) for *BACKGROUND* dataset preparation.


## Training

After the preparations, run:
```
python3 bg_colorization_main.py --mode 'train'
```

Check the logs under `outputs/(time-stamp)/log` for the changes of the losses.


## Testing

When the training finishes, the checkpoints can be found under `outputs/(time-stamp)/snapshot`. We have also provided our trained model, which can be downloaded [here](https://drive.google.com/drive/folders/1wGU3vln9Nc_Z2NV2F5nyt_2NbqDsvuRO?usp=sharing).

Then run the following command:

```
python3 bg_colorization_main.py --mode 'test' --resume_from '2019-00-00-00-00-00'
```
  - Set the time-stamp at `resume_from` to the model you want.

Results can be found under `outputs/(time-stamp)/results`.


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
```

