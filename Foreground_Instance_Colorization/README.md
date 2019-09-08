# Foreground Instance Colorization Module

This directory hosts the code and dataset for *Foreground Instance Colorization Module* in the SketchyScene Colorization system.

## Requirements
- Python 3
- Tensorflow (>= 1.3.0)
- scipy
- skimage
- PIL


## Preparations

- Please follow the instructions [here](/Foreground_Instance_Colorization/data) for *FOREGROUND* dataset preparation.
- For effective training and validation, we need to convert the raw data into tfrecord data:
    ```
    cd data_preparation
    python3 data_preparation.py
    ```
    
    and the tfrecord data will be placed under folder `data/tfrecord`.


## Training

After the preparations, run:
```
python3 obj_colorization_main.py --mode 'train'
```

Check the logs under `outputs/(time-stamp)/log` for the changes of the losses.


## Validation

When the training finishes, the checkpoints can be found under `outputs/(time-stamp)/snapshot`. We have also provided our trained model, which can be downloaded [here](https://drive.google.com/drive/folders/1wGU3vln9Nc_Z2NV2F5nyt_2NbqDsvuRO?usp=sharing).

Then run the following command for validation (edgemap colorization):

```
python3 obj_colorization_main.py --mode 'val' --resume_from '2019-00-00-00-00-00'
```
  - Set the time-stamp at `resume_from` to the model you want.

Results can be found under `outputs/(time-stamp)/validation_results`.


## Testing

Make sure the trained model has been placed correctly as mentioned in [Validation part](#validation).

Then run the following command for testing (sketch colorization):

```
python3 obj_colorization_main.py --mode 'test' --resume_from '2019-00-00-00-00-00'
```
  - Set the time-stamp at `resume_from` to the model you want.

Results can be found under `outputs/(time-stamp)/test_results`.


## Inference

Here you can select an sketch instance image and input any instructions to see the visual results. Please make sure the trained model has been placed correctly as mentioned in [Validation part](#validation).

:fire: **We have provided some wild sketches** under `examples/` folder. You can also try your own sketches. Try them as:

```
python3 obj_colorization_main.py --mode 'inference' --resume_from '2019-00-00-00-00-00' \
                                 --infer_name 'car.png' --instruction 'the car is yellow with blue window'
```
  - Set the time-stamp at `resume_from` to the model you want.
  - Set the `infer_name` to the sketch name you want.
  - Set the `instruction` to the colorization goal you want.

Results can be found under `outputs/(time-stamp)/inference_results`.


## Credits

- The code is mostly borrowed from [wchen342/SketchyGAN](https://github.com/wchen342/SketchyGAN) and [chenxi116/TF-phrasecut-public](https://github.com/chenxi116/TF-phrasecut-public).


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
