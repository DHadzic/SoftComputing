# Soft Computing project

Face detection (YOLO), emotion and gender classification (CNN).

## Requirements

Anaconda3 (keras, tensorflow - version 1.8.0, pandas, opencv, PIL)
opencv-python ( installed with pip for anaconda python interpreter using command 'pip install opencv-python )

## Weights for YOLO dnn

Pretrained weights -https://ufile.io/fvz9h  
Weights should be places inside soft-project/face_recognition directory next to .cfg files

## Training YOLO

Darket repositroy is required to train yolo configuration - https://github.com/pjreddie/darknet.  
Training data - https://ufile.io/46z1i  
Train yolo configuration, you first have to install darknet, then follow the instructions from official darknet yolo page for given training data.

## Usage instructions

For running the app through console:  
    Navigate into face_recognition folder. From there, there are several options.
    To run with web camera insert:  
    ```
    python consoleMain.py -t 0
    ```  
    To run with video input insert:  
    ```
    python consoleMain.py -t 1
    ```  
    To run with image input insert:  
    ```
    python consoleMain.py -t 2
    ```  
    For video and image input, additional console input will be required. The user will have to insert path to video/image he would like to process.   
    However, if user wants to open GUI, the following command will do that:  
    ```
    python MyWindow.py
    ``` 
