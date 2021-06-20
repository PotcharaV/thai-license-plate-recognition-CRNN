# thai-license-plate-recognition-CRNN
 
## To Generate License Plate for Training
1. Navigate to license_plate_generator folder
2. Open cmd
3. type "python Generator_augmentation -n [number of image to generate] -s save -i [directory to save image]

## To Train Model
1. Navigate to root folder
2. Open cmd
3. python training.py
** automatically save weight data in every each epoch

## To Evaluate
I use Evaluate_Predict_Showcase.ipynb to evluate the model.

## File Description
os : Ubuntu 16.04.4 LTS
Python : 3.7
Tensorflow : 2.3.0
Keras : 2.4.3

| File  | Description |
| ------------- | ------------- |
| Model.py  | Contain CRNN model  |
| parameter.py  | Contain parameters  |
| training.py  | Model training file  |
| Image_Generator.py  | Image Preprocessing and formating before .fit()  |
| save_model.py  | Saving trained model as .h5  |
| LSTM+BN5--thai-v3.hdf5  | Contain Trained CRNN weight  |
| Model_LSTM+BN5--thai-v3.h5  | Contain Trained CRNN model |  

Credit: https://github.com/qjadud1994/CRNN-Keras
## This model still need be fine-tuned with real license plate images
