# Introduction

This folder contains the code and image list files obtained from Peter on 2020/08/11 and 2020/09/29, respectively.
The following files are included in this folder

**read_Images.py**
* read the dicom image database "image_data" 
* unpack the annotaion from the overlay 
* detect the center of the annotation
* write a database with the information ('Patient_Number' 'Image_Number' 'Annotatio present' 'center_col of annotation' 'center_row of annotation' 'If origin is aligned to image' 'PhotometricInterpretation')

**train_inceptionresnetv2_V2.py**
* load the pretrained inceptionresnetv2 model from Keras
* retrain the inceptionresnetv2 model using SV dataset
* evaluate the retrained model on test data
* save the retrained model

**train_inceptionresnetv2_tensorboard.py**
* retrain the pretrained inceptionresnetv2 model on SV dataset
* record the training metrics/details using tensorboard
* evaluate the retrained model on test data
* save the retrained model

**train_inceptionresnetv2_data_pos_neg_Jan_2020.py**
* example code provided along with blog post:
 Keras InceptionResetV2 (https://jkjung-avt.github.io/keras-inceptionresnetv2/)
* train a dog vs cat classification model using Keras implemenation of InceptionResetV2 model

**train_resnet50_kk.py**
* example code provided along with blog post:Keras Cats Dogs Tutorial (https://jkjung-avt.github.io/keras-tutorial/)
* train a dog vs cat classification model using Keras implemenation of Resnet50 model

**AUC_catsdogs6.py**
* example code provided along with blog post:
Keras Cats Dogs Tutorial' (https://jkjung-avt.github.io/keras-tutorial/)
* evaluate loaded model on specified images
* calculate TP, FN, TN, FP and roc_auc
* draw ROC  curve

**newlists.zip**
* contain the imagelists of training and test sets, collected by Peter