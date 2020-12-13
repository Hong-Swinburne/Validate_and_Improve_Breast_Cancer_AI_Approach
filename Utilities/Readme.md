# Introduction

This folder contains the code of data preparation and pre-processing used for training the global framework-based and global+local framework-based mammogram classifiers.

The following files are included in this folder

**read_Images.py**
* Convert DICOM data to 'PNG' images
* Collect image information (incl. label, side, view, modality, monochrome etc.) from DICOM data and save into a csv file
* Adjust/enhance contrast for 'bright' mammograms

**image_transformations.py**
* Implement gamma/power transformation
* Implement inverse log transformation

**crop_breast.py**
* Generate binary mask for breast region
* Remove texts and the background area containing no ROI in the mammogram
  
**generate_mammo_training_test_set.py**
* Generate original SV training and test data for mammo classifier via the file lists obtained from Peter
  
**generate_patch_dataset.py**
* Generate original SV training and test data for patch classifier from the patch data obtained from Peter

**generate_kfold_data.py**
* Split the original SV data into k folds of training data (90% of the original data) and test data(10% of the test data) to validate the stability of classifier under a k-fold stratified cross-validation setting