# Final Project: Classification and Detection with Convolutional Neural Networks 

## Run Application
Make sure your terget images are placed in the folder "input", CNN weights file named "model5epochweights2_40kneg_noresize_aug20_minus_mean.h5" is placed within the same folder as the script, then execute "python run_v2.py" in terminal, the output will be saved in "graded_images".

## Train Network
Run model.py. There are 6 different models (net1 to net6), you can choose one to instantiate within model.py script, and start trainig. Different training data sets files ends with ".hdf5" are also placed in the same folder, you can choose different datasets to be used for training inside model.py script.

## Data preparation
Data files were created from SVHN .mat files, and convered to hdf5 format using conversion.py. "gs" folder should contain Google Streetview images from which the non-digit samples were generated from. "cut.py" will read files from folder "gs" and cut small patches, then save to "cutimg", "add_neg_to_file.py" will add those cut patches to converted SVHN data file.

## Model weights
Weights file "model5epochweights2_40kneg_noresize_aug20_minus_mean.h5" is used for detection. Make sure this file is in the same folder as run_v2.py. Weights obtained from other training sessions or other models are not submitted due to size limit

## Video Links
Result video: https://youtu.be/e7xge9eWlSY
Backup link: https://1drv.ms/v/s!AlUiFxjxy5SQhP5YHjsNqNVM0qX7QQ
