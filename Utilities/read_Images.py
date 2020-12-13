# This file reads the dicom image database "image_data",
# convert DICOM data into PNG image
# unpacks the annotaion from the overlay, 
# detects the center of the annotation,
# writes a database with the information ie:
# Patient_Number' 'Image_Number' 'Annotatio present' 'center_col of annotation' 'center_row of annotation' 
# 'If origin is aligned to image' 'PhotometricInterpretation')

import os
import numpy as np
import pydicom
import pandas as pd
import csv
import struct
from struct import *
import cv2
from collections import Counter
from.image_transformations import gamma_transform
from.image_transformations import inverse_log_transform

images_path = '../data/File2-Cancer/'
image_files = []
center_col = 0
center_row = 0
Monochrome2 = 1
annotation_images = 0
saveimg_bits = 8
max_val = 2**saveimg_bits - 1
# parameter for gamma/power transformation
gamma = 2.0 # need to tune this parameter for different bright images

for dirname, dirnames, filenames in os.walk(images_path):  # collectsnames and path of all images - neg and pos

    for filename in filenames:
        image_files.append(os.path.join(dirname, filename))

print('sorting file names')
image_files = [f for f in image_files if f.endswith('.dcm')]
image_files.sort()

exams = [os.path.normpath(image_file).split(os.path.sep)[-4] for image_file in image_files]
exam_counts = Counter(exams)
# print(exam_counts)
exams = list(set(exams))
print('{} exams exported from data folder'.format(len(exams)))

# load csv files for 904 malignant cases with missing annotations
df = pd.read_csv('Corrected_Cancer_exams_without_annotations_csv.csv')
cancerexams_without_label = df['Accession Number'].tolist()
sideinfo_cancerexams_without_label = df['Side'].tolist()

modality_dict = {}
views_dict = {}
lateralviews_dict = {'L_CC':0, 'L_MLO':0, 'L_OTHER':0, 'R_CC':0, 'R_MLO':0, 'R_OTHER':0}
age_dict = {}
patient_imgnum_dict = {} #dict storing the image numbers for all patients
device_dict = {} #dict storing the manufacturor machine
patient_histopathology = {} # dict storing patients' histopathology
patient_accessionnum = {} # dict storing patients' accessionnum
histopathology_imgdict = {} # dict storing image path for each histopathology
no_dcmdata = [] # list containing name of file that don't have dcm data
MGdata = [] # list containing non-MG data
annotation_images = 0 # number of annotation images
nocancer_images = []
nocancer_exams = []
nocancer_exams_histo = []
nocancer_patient = []
cancer_images = []
cancer_patient = []
#======================================================================================================================================================
# cancer images (from cancer exams without labels) whose side are confirmed in 'Corrected_Cancer_exams_without_annotations_csv.csv' as worse malignant
# These cancer images can be used as cancer samples for training/test
#=======================================================================================================================================================
cancer_images_without_label=[] 
#=====================================================================================================================================
# non-cancer images (from cancer exams without labels) whose side are listed in 'Corrected_Cancer_exams_without_annotations_csv.csv'
# These non-cancer images may not be used as normal samples for training/test, considering the bilateral malignant breasts cases
#=====================================================================================================================================
noncancer_images_without_label=[]
#======================================================================================================================================================
# Images (from cancer exams without labels) lack side information can be considered as suspicious cancer images
# These suspicious cancer images may not be used as cancer samples for training/test
#=======================================================================================================================================================
suspicious_cancer_images_without_label = []
cancer_exams = []
image_num = 0

#=======================================================================================================================#
print(len(image_files), 'dicom files')

cancer_imagelist = []
noncancer_imagelist = []
jpg_dir_name='../data/File2-Cancer-image/'
try:
    os.mkdir(jpg_dir_name)
except:
    print(os.path.exists(jpg_dir_name))

write_csvfile = True

if write_csvfile:
    with open('ImagesDB.csv', mode='w') as csv_file:
        fieldnames = ['Patient_Number', 'Accession_Number', 'Image_ID', 'Modality', 'Laterality', 'View', 'Width', 'Height', 'Annotated',
                    'center_col', 'center_row', 'BadOrigin', 'Monochrome2', 'FileName', 'FileFullDes']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(image_files)):
            print('processing {} image'.format(i+1))
            exam_ID = image_files[i].split('/')[-4]
            image_ID = image_files[i].split('/')[-2]
            jpg_image_name = exam_ID + '-' + image_ID
            bright_breast = False
            do_gamma_transform = False
            image_dcm = pydicom.read_file(image_files[i])
            IsAnnotation = False
            if 'Rows' in image_dcm:
                img_height = image_dcm.Rows
            else:
                img_height = 0
            if 'Columns' in image_dcm:
                img_width = image_dcm.Columns
            else:
                img_height = 0
            try:
                OLOrigin = image_dcm[0x6000, 0x0050].value
                IsAnnotation = True
                Overlay = 1
                if OLOrigin == [1, 1]:
                    Origin = 1
                else:
                    Origin = 2
            except:
                Overlay = 0
                Origin = 0
                center_col = 'N/A'
                center_row = 'N/A'

            BadOrigin = Origin - Overlay
            hasImage = True
            if 'PixelData' in image_dcm:
                BrImage = image_dcm.pixel_array
                img_height = BrImage.shape[0]
                img_width = BrImage.shape[1]
                if BrImage.ndim == 2 and 'PhotometricInterpretation' in image_dcm:
                    try:
                        if image_dcm.PhotometricInterpretation.endswith('1'):
                            Monochrome2 = 0
                        else:
                            Monochrome2 = 1
                    except:
                        Monochrome2 = 2
                    if 'BitsStored' in image_dcm:
                            store_bits = image_dcm.BitsStored
                            if Monochrome2 == 1:
                                monochrome = 'Mono2'
                            elif Monochrome2 == 0:
                                monochrome = 'Mono1'
                            else:
                                monochrome = 'Invalid Monochrome'
                            print(jpg_image_name, monochrome)
                            if Monochrome2 == 0:
                                # Monochrome 1
                                bright_breast = True
                                if store_bits != 14:
                                    if store_bits == 10:
                                        BrImage = 1024 - BrImage
                                    else:
                                        # 12-bit mammogram
                                        BrImage = 4095 - BrImage
                                    gamma = large_gamma
                                    if store_bits == 16:
                                        if 'WindowCenter' in image_dcm and 'WindowWidth' in image_dcm:
                                            window_center = image_dcm.WindowCenter
                                            window_width = image_dcm.WindowWidth
                                            window_center = window_center[2]
                                            window_width = window_width[0]
                                        else:
                                            do_gamma_transform = True
                                else:
                                    #==========================================
                                    # store_bits == 14
                                    # 16383 = 2**14 - 1
                                    #==========================================
                                    BrImage = 16383 - BrImage
                            elif Monochrome2 == 1:
                                # Monochrome 2
                                if store_bits == 12:
                                    if 'WindowCenter' in image_dcm and 'WindowWidth' in image_dcm:
                                        window_center = image_dcm.WindowCenter
                                        window_width = image_dcm.WindowWidth
                                        if window_center < 2047:
                                            #  bright breasts
                                            bright_breast = True
                            if bright_breast:
                                width = img_width
                                height = img_height
                                transform_img = np.zeros((height, width), dtype=np.uint8)
                                if not do_gamma_transform:
                                    # do piecewise linear intensity transformation
                                    a = int(window_center-window_width/2)
                                    b = int(window_center+window_width/2)
                                    for i in range(height):
                                        for j in range(width):
                                            r = BrImage[i,j]
                                            if r < a:
                                                transform_img[i,j]=0
                                            elif r > b:
                                                transform_img[i,j]=max_val
                                            else:
                                                transform_img[i,j]=int((r-a)/window_width*max_val)
                                else:
                                    # do gamma/power transform
                                    transform_img = gamma_transform(BrImage, gamma, BrImage.max())
                                    # transform_img = inverse_log_transform(BrImage, BrImage.max())
                                BrImage = transform_img

                BrImage = BrImage / max_val
                BrImage = cv2.normalize(BrImage, BrImage, 0, max_val, cv2.NORM_MINMAX)
                if saveimg_bits == 8:
                    BrImage = np.uint8(BrImage)
                else:
                    # 16-bit
                    BrImage = np.uint16(BrImage)
                # save png image converted from DICOM
                cv2.imwrite(os.path.join(jpg_dir_name, jpg_image_name+'.png'), BrImage)
                print('writing {}'.format(jpg_image_name+'.png'))

            else:
                hasImage = False

            if IsAnnotation:
                annotation_images += 1
                # decode overlay array into an image - OlayImage
                rows = image_dcm[0x6000, 0x0010].value
                cols = image_dcm[0x6000, 0x0011].value
                n_bits = 8
                OLImage = image_dcm[0x6000, 0x3000].value

                Max_roi_diam = 70

                if BadOrigin == 0:  # annotated with origin at [1,1]
                    UnpackOK = False
                    PrintOK = False
                    AllFailed = False
                    # print(image_files[i], 'i= ', i)

                    bb = struct.unpack_from('B' * len(OLImage), OLImage, 0)  # unpack annotation data into bytes
                    bba = np.array(bb)
                    cc0 = np.unpackbits(bba.astype(np.uint8))  # unpack annotation data into bits

                    # print('integer unpack ', image_files[i][52:65], ' i = ', i, 'check that ', len(cc0), ' = ',
                    #         (rows * cols))
                    cc = cc0[0:rows * cols]
                    cd = np.reshape(cc, [rows, cols])

                    center_row = 0
                    center_col = 0
                    c = 10

                    # determine center of annotation
                    while c < cols and center_row == 0:
                        if np.sum(cd[:, c]) > 0:  # find left side of annotation
                            # print('left col = ', c)
                            list_c = list(cd[:, c])
                            rev_list_c = list_c[::-1]
                            center_row1 = list_c.index(1)  # find center row of annotation
                            center_row2 = rows - rev_list_c.index(1)
                            center_row = int((center_row1 + center_row2) / 2)
                            # print('center_row = ', center_row)
                            Left_col = c
                            still_searching = False
                        c += 1

                    center_col = 0
                    r = 0
                    while r < rows and center_col == 0:
                        if np.sum(cd[r, :]) > 0:  # find left side of annotation

                            # print('upper row = ', r)
                            list_r = list(cd[r, :])
                            rev_list_r = list_r[::-1]
                            center_col1 = list_r.index(1)  # find center row of annotation
                            center_col2 = cols - rev_list_r.index(1)
                            center_col = int((center_col1 + center_col2) / 2)
                            # print('center_col = ', center_col)
                            Top_r = r
                        r += 1
         
            full_desc = jpg_image_name
            if 'PatientID' in image_dcm:
                PatientID = image_dcm.PatientID
            else:
                PatientID = 'N/A'
            if 'AccessionNumber' in image_dcm:
                AccessionNumber = image_dcm.AccessionNumber
            else:
                AccessionNumber = 'N/A'
            if 'ImageLaterality' in image_dcm:
                ImageLaterality = image_dcm.ImageLaterality
                full_desc = full_desc+'-'+ImageLaterality
            else:
                ImageLaterality = 'N/A'
            if 'ViewPosition' in image_dcm:
                ViewPosition = image_dcm.ViewPosition
                full_desc = full_desc+'-'+ViewPosition
            else:
                ViewPosition = 'N/A'
            if 'Modality' in image_dcm:
                Modality = image_dcm.Modality
                full_desc = full_desc+'-'+Modality
            else:
                Modality = 'N/A'
                
            if IsAnnotation:
                full_desc = full_desc + '-cancer'
                cancer_imagelist.append(full_desc)
            else:
                # if exam_ID is in cancer cases without label and the image side matches the labeled side of cancer cases in 
                # 'Corrected_Cancer_exams_without_annotations_csv.csv', mark it as cancer image and move it to cancer_images_without_label.
                # If the image side does not match the labeled side of cancer cases in 'Corrected_Cancer_exams_without_annotations_csv.csv',
                # mark it as suspicious_nocancer, move it to noncancer_images_without_label
                # If the image lacks side info, mark it as suspicious_cancer, move it to suspicious_cancer_images_without_label
                if exam_ID in cancerexams_without_label:
                    idx = cancerexams_without_label.index(exam_ID)
                    side = sideinfo_cancerexams_without_label[idx]
                    if side == ImageLaterality:
                        full_desc = full_desc + '-cancer'
                        cancer_images_without_label.append(full_desc)
                    elif ImageLaterality is not 'N/A':
                        full_desc = full_desc + '-suspicious_nocancer'
                        noncancer_images_without_label.append(full_desc)
                    else:
                        full_desc = full_desc + '-suspicious_cancer'
                        suspicious_cancer_images_without_label.append(full_desc)
                else:
                    full_desc = full_desc + '-nocancer'
                    noncancer_imagelist.append(full_desc)
            
            if Modality == 'MG':
                MGdata.append(full_desc)

            if write_csvfile:
                writer.writerow(
                    {'Patient_Number': image_dcm.PatientID, 'Accession_Number': AccessionNumber, 'Image_ID': image_ID, 'Modality': Modality, 'Laterality': ImageLaterality,
                    'View': ViewPosition, 'Width': img_width, 'Height': img_height, 'Annotated': Overlay,
                    'center_col': center_col, 'center_row': center_row, 'BadOrigin': BadOrigin, 'Monochrome2': Monochrome2, 'FileName':jpg_image_name, 
                    'FileFullDes':full_desc})

#output cancer and nocancer 
with open('cancer_imagelist.txt', mode='w') as f:
    for img_name in cancer_imagelist:
        f.write(img_name + os.linesep)
f.close()
with open('nocancer_imagelist.txt', mode='w') as f:
    for img_name in noncancer_imagelist:
        f.write(img_name + os.linesep)
f.close()

with open('cancer-without-label_imagelist.txt', mode='w') as f:
    for img_name in cancer_images_without_label:
        f.write(img_name + os.linesep)
f.close()

with open('suspicious-nocancer-without-label_imagelist.txt', mode='w') as f:
    for img_name in noncancer_images_without_label:
        f.write(img_name + os.linesep)
f.close()

with open('suspicious-cancer-without-label_imagelist.txt', mode='w') as f:
    for img_name in suspicious_cancer_images_without_label:
        f.write(img_name + os.linesep)
f.close()

with open('MG_imagelist.txt', mode='w') as f:
    for img_name in MGdata:
        f.write(img_name + os.linesep)
f.close()

print(len(image_files), 'dicom files')
print('%d has annotations'%(annotation_images))
