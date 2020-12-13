
# This file reads the dicom image database "image_data", 
# unpacks the annotaion from the overlay, 
# detects the center of the annotation,
# writes a database with the information ie:
# Patient_Number' 'Image_Number' 'Annotatio present' 'center_col of annotation' 'center_row of annotation' 'If origin is aligned to image' 'PhotometricInterpretation')

import os
import numpy as np
import scipy.io as sio
from scipy import misc
import scipy
import pydicom
import pandas
import csv
import struct
from struct import *
from PIL import Image




images_path = '../image_data/'
image_files=[]
center_col=0
center_row=0
Monochrome2=1

for dirname, dirnames, filenames in os.walk(images_path):   # collectsnames and path of all images - neg and pos

    for filename in filenames:
        image_files.append(os.path.join(dirname, filename))

print('sorting file names')
image_files = [f for f in image_files if f.endswith('.dcm')]
image_files.sort()

print(len(image_files), 'dicom files')

jpg_dir_name='overlay_jpgs/'
try:
    os.mkdir(jpg_dir_name)
except:
    print(os.path.exists(jpg_dir_name))


with open('ImagesDB.csv', mode='w') as csv_file:
	fieldnames = ['Patient_Number', 'Image_Number', 'Annotated', 'center_col', 'center_row', 'BadOrigin', 'Monochrome2']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	writer.writeheader()
	for i in range(len(image_files)):
		image_dcm = pydicom.read_file(image_files[i])
		IsAnnotation=False
		try:
			OLOrigin=image_dcm[0x6000, 0x0050].value
			IsAnnotation=True
			Overlay=1
			if OLOrigin==[1,1]:
				Origin=1
			else:
				Origin=2
		except:
			Overlay=0
			Origin=0

		BadOrigin=Origin-Overlay

		if IsAnnotation:
			# decode overlay array into an image - OlayImage
			rows=image_dcm[0x6000, 0x0010].value
			cols=image_dcm[0x6000, 0x0011].value
			n_bits=8
			OLImage=image_dcm[0x6000, 0x3000].value

			Max_roi_diam=70

			if BadOrigin==0:  # annotated with origin at [1,1]
				UnpackOK=False
				PrintOK=False
				AllFailed=False
				print(image_files[i], 'i= ',i)
			
				bb=struct.unpack_from('B'*len(OLImage), OLImage, 0)   	# unpack annotation data into bytes
				bba=np.array(bb)
				cc0=np.unpackbits(bba.astype(np.uint8)) 				# unpack annotation data into bits

				print('integer unpack ', image_files[i][52:65], ' i = ',i, 'check that ', len(cc0), ' = ', (rows*cols))
				cc=cc0[0:rows*cols]
				cd=np.reshape(cc,[rows,cols])

				center_row=0
				center_col=0
				c=10

				# determine center of annotation
				while c<cols and center_row==0:
					if np.sum(cd[:,c])>0:      	# find left side of annotation
						print('center c = ',c)
						list_c=list(cd[:,c])
						rev_list_c=list_c[::-1]
						center_row1=list_c.index(1)   # find center row of annotation
						center_row2=rows-rev_list_c.index(1) 
						center_row=int((center_row1+center_row2)/2)
						print('center_row = ',center_row)
						Left_col=c
						still_searching=False
					c+=1

				center_col=0
				r=0
				while r<rows and center_col==0:
					if np.sum(cd[r,:])>0:      	# find left side of annotation

						print('center r = ',r)
						list_r=list(cd[r,:])
						rev_list_r=list_r[::-1]
						center_col1=list_r.index(1)   # find center row of annotation
						center_col2=cols-rev_list_r.index(1) 
						center_col=int((center_col1+center_col2)/2)
						print('center_col = ',center_col)
						Top_r=r
					r+=1

				OlayImage=cd
				print('OlayImage.shape =', OlayImage.shape, 'center_col = ',center_col, 'center_row =', center_row)
				OlayImage[:,center_col]=1
				OlayImage[center_row,:]=1

				PrintOK=True

				if PrintOK:	

					BrImage=image_dcm.pixel_array
					BrImage=BrImage/65535
					if BrImage.ndim==2:
						PrintOK=True
					else:
						PrintOK=False

				if PrintOK:	
					try:
						if image_dcm.PhotometricInterpretation.endswith('1'):				
							BrImage=4095-BrImage
							BrImage=BrImage*2
							print('MONOCHROME1')
							Monochrome2=0
						else:
							Monochrome2=1
					except:
							Monochrome2=2

					if BrImage.shape ==	OlayImage.shape:			
						BrOlay=BrImage+OlayImage
					else:
						[Br_y,Br_x]=BrImage.shape
						OlayImage=OlayImage[:Br_y,:Br_x]
						print('BrImage.shape = ', BrImage.shape)
						print('OlayImage.shape = ', OlayImage.shape)
						BrOlay=BrImage+OlayImage

					#jpg_image_name=image_files[i][19:32]+'_im_'+image_files[i][38]+'.jpg'
					jpg_image_name=image_files[i][52:65]+'_im_'+image_files[i][71:72]
					if np.max(OlayImage)<=1:
						OlayImage=OlayImage*255
					scipy.misc.toimage(OlayImage).save(jpg_dir_name+jpg_image_name+'_annotation.jpg')
					scipy.misc.toimage(BrImage).save(jpg_dir_name+jpg_image_name+'_image.jpg')
					scipy.misc.toimage(BrOlay).save(jpg_dir_name+jpg_image_name+'_overlay.jpg')


		# writer.writerow({'Patient_Number': image_files[i][19:32], 'Image_Number': image_files[i][38], 'Annotated': Overlay, 'center_col': center_col, 'center_row': center_row, 'BadOrigin': BadOrigin, 'Monochrome2': Monochrome2})
		writer.writerow({'Patient_Number': image_files[i][52:65], 'Image_Number': image_files[i][71:72], 'Annotated': Overlay, 'center_col': center_col, 'center_row': center_row, 'BadOrigin': BadOrigin, 'Monochrome2': Monochrome2})



# image = image_dcm.pixel_array
# plt.imshow(image)
# plt.show()
# OLOrigin=image_dcm.OverlayOrigin_0
