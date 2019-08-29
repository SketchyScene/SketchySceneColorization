# *BACKGROUND* Dataset Preparation

Please follow these instructions for data preparation for the *Background Colorization Module*.

1. Download the image data (foreground/user_paint/inner_mask) [here](https://drive.google.com/drive/folders/1U9u26q6W-HyUXRn_Ppkz3vXPstq7-ebN?usp=sharing), and put them under this folder following this structure:

    ```
    foreground/
    user_paint/
    inner_mask/
    ......
    
    ```
    
1. As the whole dataset is too large (~6G), we have only provided the slim version of the dataset (necessary files in train & test set) and recommend users to generate the whole dataset (with augmentation) by their own. And we have provided the code in `data_preparation` directory. Run:

    ```
    cd ../data_preparation
    python3 bg_data_generation.py
    ```

    Afterwards, you will see the whole dataset with the following folder structures:
    ```
    foreground/
    user_paint/
    inner_mask/
    
    background/
    captions/
    segment/
    ......
    
    ```
