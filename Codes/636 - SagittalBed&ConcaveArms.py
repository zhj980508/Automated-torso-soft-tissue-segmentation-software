# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 15:43:07 2021

@author: Huijing Zhou
"""

# In[]:Introduction --- Arm removal experiment

## Aim: Arm removal
# Automated seed point searching algorithm: Sitable for left/right arm sides
# Methods: Find the concave points and cut the left and right side directly

# In[]: Import the libs

import cv2 as cv
import numpy as np
import os
import pydicom
from matplotlib import pyplot as plt
from skimage import measure,morphology
import scipy
from scipy.signal import find_peaks
import time

# In[]: Functions written by Huijing

def FindAllTargetSlices(dir,SliceNumber):
    
    results = []
    folders = [dir]
    
    for i in range(len(SliceNumber)):
        specify_str = SliceNumber[i]
        for folder in folders :
            folders += [os.path.join(folder, x) for x in os.listdir(folder) \
                        if os.path.isdir(os.path.join(folder, x))]
            results += [os.path.relpath(os.path.join(folder, x), start = dir) \
                        for x in os.listdir(folder) \
                        if os.path.isfile(os.path.join(folder, x)) and specify_str in x]
                
    return results


def FindOriginalDataByVertebraName(VertebraName,Vertebra_Filename_list,dir):
    
    for i in range(len(Vertebra_Filename_list)):
        if Vertebra_Filename_list[i][0] == VertebraName:
            file_name = Vertebra_Filename_list[i][1]
        else: i += 1
        
    file_path = os.path.join(dir,file_name)
    ds = pydicom.dcmread(file_path)
    data = ds.pixel_array 
     
    return ds, data


def GrayscaleToHU(ds,data):
    # Function: Grayscale to HU
    
    # Rescale Intercept
    RescaleIntercept = ds[0X0028, 0X1052].value
    # Rescale Slope
    RescaleSlope = ds[0X0028, 0X1053].value
    # HU = pixel * slope + intercept
    data_HU = data * RescaleSlope + RescaleIntercept
    
    return data_HU
    

def SetContrastInCT(data_HU,HU_min,HU_max):
    
    # Function: Set contrast in CT
    # bone contrast: 250 HU to 1000 HU
    # soft tissue contrast: -150 HU to 350 HU
    
    slope = 255 / (HU_max - HU_min)
    intercept = -slope * HU_min
    
    data_Contrast = np.round(slope * data_HU + intercept)
    data_Contrast[data_Contrast < 0] = 0
    data_Contrast[data_Contrast > 255] = 255
    
    return data_Contrast


def LoadImageData_HU_SoftTissue(VertebraName,Vertebra_Filename_list,dir):
    
    
    ds,data = FindOriginalDataByVertebraName(VertebraName,Vertebra_Filename_list,dir)
    data_HU = GrayscaleToHU(ds, data)
    SoftTissueHU_min = -150
    SoftTissueHU_max = 350
    data_HU_SoftTissue = SetContrastInCT(data_HU,SoftTissueHU_min,SoftTissueHU_max)
    
    return data_HU_SoftTissue


def CTResize_Resolution(data,PixelSize=0.976562): #### PixelSize = 0.976562mm
    
    height, width = data.shape[0], data.shape[1]
    height_res = int(height * PixelSize)
    width_res = int(width * PixelSize)
    dim = (width_res,height_res)
    data_res = cv.resize(data,dim,interpolation = cv.INTER_AREA)
    
    return data_res


def InitialTorsoSeg(data_HU_SoftTissue,kernel):
    # Function: Initial Torso Region Segmentation
    # Parameters:
    # data_HU_SoftTissue:data need to be segmented
    # kernel:close operation kernel
    
    ret,mask = cv.threshold(data_HU_SoftTissue,0,255,cv.THRESH_BINARY) 
    morph_close = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) # close operation to remove small spots
    labels, num=measure.label(morph_close,connectivity=2,background=0,return_num=True) # label connected areas
    
    NumberofEachArea = []
    for i in range(num+1):
        NumberofEachArea.append(np.sum(labels == i))
        
    tmp_list=sorted(NumberofEachArea) # sort the labels
    final = tmp_list[-2] # The most pixels are in the background, so choose the second largest label after sorting
    idx = NumberofEachArea.index(final) # label we want
    MaxConnectedArea = np.where(labels == idx, 1, 0) # keep all the pixels that label=idx
    Torso_initial_mask = scipy.ndimage.binary_fill_holes(MaxConnectedArea)
    
    return  Torso_initial_mask,MaxConnectedArea


def BodyCavitySeg(Torso_initial_mask,MaxConnectedArea,min_size):
    # Function: Body cavity segmentation
    # Parameters:
    # Torso_initial_mask: from InitialTorsoSeg
    # MaxConnectedArea: from InitialTorsoSeg
    # min_size: The smallest tolerated area of the body cavity
    
    BodyCavity_initial = Torso_initial_mask - MaxConnectedArea
    label_BodyCavity = measure.label(BodyCavity_initial) # Zoning of areas of initial soft tissue, 8 contiguous
    BodyCavity_mask = morphology.remove_small_objects(label_BodyCavity, min_size, connectivity=2) # remove small objects with the min_size
    BodyCavity_mask = BodyCavity_mask > 0 # label has values, so the returned mask needs to be binarized, leaving only 0, 1
    
    return BodyCavity_mask


def ArmRemovalBySearchingConcavePoints(Torso_initial_mask,height,distance):
    # Function: Return Torso_mask
    # Torso_initial_mask
    # distance: a approximate number of points between left and right arms, usually 200
    # height: usually 100
    
    # Sum all the rows of Torso_initial_mask threshold the profile
    Torso_initial_mask_sum_rows = np.sum(Torso_initial_mask, axis=0)

    # negative uint32 is a bit strange, so there convert uint32 to float64
    Torso_initial_mask_sum_rows = np.array(list(map(np.float,Torso_initial_mask_sum_rows)))
    Torso_initial_mask_sum_neg = -Torso_initial_mask_sum_rows + np.max(Torso_initial_mask_sum_rows)
    plt.plot(Torso_initial_mask_sum_neg)

    # find peak locations -- find concave points
    peak_locx,peak_property = find_peaks(Torso_initial_mask_sum_neg, height=100, distance=200,width=0.6)
    peak_height = peak_property['peak_heights']
    print('peak_locx',peak_locx)
    print('peak_height',peak_height) 

    NumofConcavePoints = len(peak_locx) 
    
    Torso_mask = Torso_initial_mask   
    
    if NumofConcavePoints == 0:
        Torso_mask = Torso_initial_mask
    elif NumofConcavePoints == 1 and peak_locx[0] < 0.5 * Torso_initial_mask.shape[1]: # Left Arm
        Arm_LeftConcave = peak_locx[0]
        Torso_mask[:,0:Arm_LeftConcave] = 0
    elif NumofConcavePoints == 1 and peak_locx[0] > 0.5 * Torso_initial_mask.shape[1]: # Right Arm
        Arm_RightConcave = peak_locx[0]
        Torso_mask[:,Arm_RightConcave:Torso_mask.shape[1]] = 0
    elif  NumofConcavePoints == 2: # Left & Right Arms
        Arm_LeftConcave = peak_locx[0]
        Arm_RightConcave = peak_locx[1]
        Torso_mask[:,0:Arm_LeftConcave] = 0
        Torso_mask[:,Arm_RightConcave:Torso_mask.shape[1]] = 0
    else:
        Torso_mask = Torso_initial_mask
        
    return Torso_mask


def AllAreaPerimeterCentroid(contours):  
    
    area = np.zeros(len(contours))
    perimeter = 0
    cx = np.zeros(len(contours))
    cy = np.zeros(len(contours))
    
    for i in range(len(contours)):
        perimeter += cv.arcLength(contours[i],True)
        area[i] = cv.contourArea(contours[i])
        M = cv.moments(contours[i])
        cx[i] = int((M['m10']+1)/(M['m00']+1))
        cy[i] = int((M['m01']+1)/(M['m00']+1))
        #cx[i] = int(M['m10']/M['m00'])
        #cy[i] = int(M['m01']/M['m00'])

    return area,perimeter,cx,cy


def SoftTissueParameter(contours_Torso,contours_BodyCavity):
    
    # calculate parameters of torso
    Area_Torso, Perimeter_Torso,cx_Torso,cy_Torso = AllAreaPerimeterCentroid(contours_Torso)
    
    # calculate parameters of body cavity
    Area_BodyCavity, Perimeter_BodyCavity,cx_BodyCavity,cy_BodyCavity = AllAreaPerimeterCentroid(contours_BodyCavity)
   
    # calculate parameters of soft tissue
    Area_SoftTissue = np.sum(Area_Torso) - np.sum(Area_BodyCavity)
    Perimeter_SoftTissue = np.sum(Perimeter_Torso) + np.sum(Perimeter_BodyCavity)
    cx_SoftTissue =  (cx_Torso * Area_Torso - np.sum(cx_BodyCavity * Area_BodyCavity)) / Area_SoftTissue  
    cy_SoftTissue =  (cy_Torso * Area_Torso - np.sum(cy_BodyCavity * Area_BodyCavity)) / Area_SoftTissue                        

    return Area_SoftTissue,Perimeter_SoftTissue,cx_SoftTissue,cy_SoftTissue


def dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


# In[]: Load all the data that needs to be processed

# 606 from C2 to L5

dir_606 = r'..\CT Scans\00606M_10_18_12\Dicom'
VertebraName_606 = ['C2','C3','C4','C5','C6','C7',
                    'T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12',
                    'L1','L2','L3','L4','L5']
SliceNumber_606 = ['1262','1227','1198','1167','1139','1112','1084','1053','1022','0989',
               '0956','0916','0879','0841','0801','0761','0716','0669','0621','0568',
               '0512','0462','0404']
Filename_606 = FindAllTargetSlices(dir_606,SliceNumber_606)
Vertebra_Filename_606 = zip(VertebraName_606,Filename_606)
Vertebra_Filename_606_list = list(Vertebra_Filename_606)    
    
# 636 from T3 to L5

dir_636 = r'..\CT Scans\00636M_03_28_18\Dicom'
VertebraName_636 = ['T3','T4','T5','T6','T7','T8','T9','T10','T11','T12',
                    'L1','L2','L3','L4','L5']
SliceNumber_636 = ['1033','1000','0962','0927','0891','0853','0813','0773','0730',
                   '0685','0637','0587','0538','0487','0433']
Filename_636 = FindAllTargetSlices(dir_636,SliceNumber_636)
Vertebra_Filename_636 = zip(VertebraName_636,Filename_636)
Vertebra_Filename_636_list = list(Vertebra_Filename_636)

# 526 from C1 to L5

dir_526 = r'..\CT Scans\00526M\Dicom'
VertebraName_526 = ['C1','C2','C3','C4','C5','C6','C7',
                    'T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12',
                    'L1','L2','L3','L4','L5']
SliceNumber_526 = ['111079','111058','111022','110990','110962','110937','110912','110884','110854',
                   '110826','110786','110748','110709','110668','110628','110585','110540','110492',
                   '110442','110391','110329','110267','110209','110150']
Filename_526 = FindAllTargetSlices(dir_526,SliceNumber_526)
Vertebra_Filename_526 = zip(VertebraName_526,Filename_526)
Vertebra_Filename_526_list = list(Vertebra_Filename_526)

# In[]: CT bed removal -- no need for choosing one slice for removal -- using sagittal plane
# Table top profile generation
start = time.time()
# Reformat the data to get a sagittal image
# load the DICOM files 

def LoadAllDicomSlicesFromOneFolder(filepath):
    files = []
    print(len(os.listdir(filepath)))
    for file in os.listdir(filepath):
        child = os.path.join(filepath, file)
        files.append(pydicom.dcmread(child))
    return files

def CTTableTopInterceptFinding(SagittalImage_HU):
    SagittalImage_HU_Binary = SagittalImage_HU > -500
    SagittalImage_HU_Binary = SagittalImage_HU_Binary.astype(np.uint8)
    plt.imshow(SagittalImage_HU_Binary,cmap='gray')
    plt.show()

    # Edge detection using Sobel
    x = cv.Sobel(SagittalImage_HU_Binary,cv.CV_16S,1,0)
    y = cv.Sobel(SagittalImage_HU_Binary,cv.CV_16S,0,1)
    absX = cv.convertScaleAbs(x)   
    absY = cv.convertScaleAbs(y)

    # Detect vertical edges by Hough transform with angle and intercept
    SagittalImage_Edges = cv.addWeighted(absX,0,absY,1,0)
    SagittalImage_Lines = cv.HoughLines(SagittalImage_Edges,1,np.pi/180,400) #400 can be changed here, you can try 300,400,500,600

    # Choose the angle that gives the largest bin count
    rho_CTbed = []
    for i in range(0,len(SagittalImage_Lines)):
        rho,theta = SagittalImage_Lines[i][0][0],SagittalImage_Lines[i][0][1]
        TolerentRad = 0.1 * np.pi/180
        if theta < 90 * np.pi/180 + TolerentRad and theta > 90 * np.pi/180 - TolerentRad:
            rho_CTbed.append(rho)
    rho_CTbed = np.sort(rho_CTbed)
    print(rho_CTbed)

    # Pick the corresponding intercept as table top
    TableTop_initial = []
    for i in range(len(rho_CTbed)):
        if rho_CTbed[i] > 0.5*SagittalImage_HU.shape[0]:
            TableTop_initial.append(rho_CTbed[i])   
    TableTop_CTbed = int(min(TableTop_initial))
    
    return TableTop_CTbed


files = LoadAllDicomSlicesFromOneFolder(dir_636)
print("file count: {}".format(len(files)))         

# skip files with no SliceLocation (eg scout views)
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1
print("skipped, no SliceLocation: {}".format(skipcount))

# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)

# pixel aspects, assuming all slices are the same
#ps = slices[0].PixelSpacing
#ss = slices[0].SliceThickness
#ax_aspect = ps[1]/ps[0]
#sag_aspect = ps[1]/ss
#cor_aspect = ss/ps[0]
ri = slices[0].RescaleIntercept
rs = slices[0].RescaleSlope

# create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))
img3d = np.zeros(img_shape)

# fill 3D array with the images from the files
for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d

# plot 3 orthogonal slices
TableTop_CTbed = np.ones(int(img_shape[1]))*img_shape[0]
# TableTop_CTbed = np.zeros(int(img_shape[1]))
for i in range(30,int(img_shape[1]-1)-30):        
    SagittalImage = img3d[:, i, :]
    SagittalImage_HU = SagittalImage * rs + ri
    TableTop_CTbed[i] = CTTableTopInterceptFinding(SagittalImage_HU)
    #AxialImage = img3d[:, :, img_shape[2]//2]
    #plt.imshow(img3d[:, img_shape[1]//2, :]) #sagittal
    #plt.imshow(img3d[:, :, img_shape[2]//2]) #axial
    #plt.imshow(img3d[img_shape[0]//2, :, :].T) #coronal
plt.plot(TableTop_CTbed)
plt.show()

# In[]: Process the data from database 606

# The target slice of 606 for parameter calculation 

VertebraName = 'L1'
data_HU_SoftTissue = LoadImageData_HU_SoftTissue(VertebraName,Vertebra_Filename_636_list,dir_636)
plt.imshow(data_HU_SoftTissue, cmap='gray')
plt.title('Original image of T8(606)')
plt.show()

# In[]: Process the data from database 606

data_CTBedRemoval = data_HU_SoftTissue
for i in range(len(TableTop_CTbed)):
    data_CTBedRemoval[int(TableTop_CTbed[i]):data_HU_SoftTissue.shape[0],i] = 0
plt.imshow(data_CTBedRemoval)
plt.plot(TableTop_CTbed)

end = time.time()
print('running time'+str(end-start))
# In[]：Intial Torso, Body Cavity, Soft TIssue Segmentation

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,1)) # the unit of close operation: circle(r=2)
Torso_initial_mask,MaxConnectedArea = InitialTorsoSeg(data_CTBedRemoval,kernel)
min_size = 15
BodyCavity_initial_mask = BodyCavitySeg(Torso_initial_mask,MaxConnectedArea,min_size)

# In[]: Arm removal -- concave point method

def ArmRemovalBySearchingConcavePoints(Torso_initial_mask,height,distance):
    # Function: Return Torso_mask
    # Torso_initial_mask
    # distance: a approximate number of points between left and right arms, usually 200
    # height: usually 100
    
    # Sum all the rows of Torso_initial_mask threshold the profile
    Torso_initial_mask_sum_rows = np.sum(Torso_initial_mask, axis=0)

    # negative uint32 is a bit strange, so there convert uint32 to float64
    Torso_initial_mask_sum_rows = np.array(list(map(np.float,Torso_initial_mask_sum_rows)))
    Torso_initial_mask_sum_neg = -Torso_initial_mask_sum_rows + np.max(Torso_initial_mask_sum_rows)
    plt.plot(Torso_initial_mask_sum_neg)

    # find peak locations -- find concave points
    peak_locx,peak_property = find_peaks(Torso_initial_mask_sum_neg, height=50, distance=200,width=0.6,prominence=1)
    peak_height = peak_property['peak_heights']
    print('peak_locx',peak_locx)
    print('peak_height',peak_height) 

    NumofConcavePoints = len(peak_locx) 
    
    Torso_mask = Torso_initial_mask   
    
    if NumofConcavePoints == 0:
        Torso_mask = Torso_initial_mask
    elif NumofConcavePoints == 1 and peak_locx[0] < 0.5 * Torso_initial_mask.shape[1]: # Left Arm
        Arm_LeftConcave = peak_locx[0]
        Torso_mask[:,0:Arm_LeftConcave] = 0
    elif NumofConcavePoints == 1 and peak_locx[0] > 0.5 * Torso_initial_mask.shape[1]: # Right Arm
        Arm_RightConcave = peak_locx[0]
        Torso_mask[:,Arm_RightConcave:Torso_mask.shape[1]] = 0
    elif  NumofConcavePoints == 2: # Left & Right Arms
        Arm_LeftConcave = peak_locx[0]
        Arm_RightConcave = peak_locx[1]
        Torso_mask[:,0:Arm_LeftConcave] = 0
        Torso_mask[:,Arm_RightConcave:Torso_mask.shape[1]] = 0
    else:
        Torso_mask = Torso_initial_mask
        
    return Torso_mask

Torso_mask = ArmRemovalBySearchingConcavePoints(Torso_initial_mask,height=50,distance=200)
#plt.imshow(Torso_mask, cmap='gray')
#plt.show()

# In[]: Torso/Body Cavity Contours

# Torso contours
Torso_mask = Torso_mask.astype(np.uint8)
ret, binary = cv.threshold(Torso_mask,0,255,cv.THRESH_BINARY)  
contours_Torso, hierarchy_Torso = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 

# Body Cavity contours
BodyCavity_mask = Torso_mask & BodyCavity_initial_mask
BodyCavity_mask = BodyCavity_mask.astype(np.uint8)
ret, binary = cv.threshold(BodyCavity_mask,0,255,cv.THRESH_BINARY)  
contours_BodyCavity, hierarchy_BodyCavity = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
 
# In[]: Parameters(area,centroid,perimeter) calculation
    
Area_SoftTissue,Perimeter_SoftTissue,cx_SoftTissue,cy_SoftTissue = SoftTissueParameter(contours_Torso, contours_BodyCavity)

print('The area of Soft Tissue is',Area_SoftTissue)
print('The perimeter of Soft Tissue is',Perimeter_SoftTissue)
print('The centroid of Soft Tissue is ',cx_SoftTissue,',',cy_SoftTissue)

# In[]: For Evalution -- load the corresponding marked image (masked by Huijing)

# Just one slice

dir_636_Markers = r'..\CT Scans\00636M_03_28_18\Markers'
ds_eva,data_eva = FindOriginalDataByVertebraName(VertebraName,Vertebra_Filename_636_list,dir_636_Markers)
data_HU_eva = GrayscaleToHU(ds_eva, data_eva)
plt.imshow(data_HU_eva, cmap='gray')
plt.title('Original image of for eva T8(606)')
plt.show()

TargetRegion_mask_eva = np.zeros([data_HU_eva.shape[0],data_HU_eva.shape[1]])
TargetRegion_mask_eva[np.where(data_HU_eva == np.max(data_HU_eva))] = True;
TargetRegion_mask_eva = TargetRegion_mask_eva.astype(np.uint8)
plt.imshow(TargetRegion_mask_eva, cmap='gray')
plt.title('TargetRegion_mask_eva')
plt.show()

# In[]: For evaluation -- Contours drawing

# Target Region contours
ret, binary = cv.threshold(TargetRegion_mask_eva,0,255,cv.THRESH_BINARY)  
contours_TargetRegion_eva, hierarchy_TargetRegion_eva = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 

# Target Region contour drawing
data_HU_SoftTissue_Color = cv.cvtColor(data_HU_SoftTissue.astype(np.uint8), cv.COLOR_GRAY2RGB)
cv.drawContours(data_HU_SoftTissue_Color,contours_TargetRegion_eva,-1,(0,255,255),1)  
# Soft Tissue contour drawing  
cv.drawContours(data_HU_SoftTissue_Color,contours_Torso,-1,(0,0,255),1)    
cv.drawContours(data_HU_SoftTissue_Color,contours_BodyCavity,-1,(0,0,255),1)  

plt.imshow(data_HU_SoftTissue_Color, cmap='gray')
plt.title('Soft tissue contour (Target Region)')
plt.show()
cv.imshow('Soft tissue_T8_00606M (Target Region)', data_HU_SoftTissue_Color)  
cv.waitKey(0)

# In[]: For Evalution -- Dice coefficient

SoftTissue_mask = Torso_mask & ~BodyCavity_mask
SoftTissue_mask = SoftTissue_mask.astype(np.uint8)
plt.imshow(SoftTissue_mask, cmap='gray')
plt.title('SoftTissue_mask')
plt.show()

DC = dice_coefficient(SoftTissue_mask,TargetRegion_mask_eva)
print('The Dice coefficient is ',DC)

Area_Target = np.sum(TargetRegion_mask_eva)
Area_SoftTissue = np.sum(SoftTissue_mask)
AreaPercentage = min(Area_Target,Area_SoftTissue)/max(Area_Target,Area_SoftTissue)
print('The Area Percentatge is ',AreaPercentage)


# In[]: Unit convert for area

PixelSize = 0.976562  # mm
Area_SoftTissue_mm = np.sum(SoftTissue_mask) * PixelSize * PixelSize # mm^2
print('Area_SoftTissue_mm ',Area_SoftTissue_mm)

Area_Torso_mm = np.sum(Torso_mask) * PixelSize * PixelSize
print('Area_Torso_mm ',Area_Torso_mm)

# In[]: Show the results

point_size = 1
point_color = (0, 255, 0) # BGR
thickness = 4
cv.circle(data_HU_SoftTissue_Color,(int(cx_SoftTissue),int(cy_SoftTissue)),point_size, point_color, thickness)
         
cv.putText(data_HU_SoftTissue_Color,'The area of Soft Tissue is '+str(Area_SoftTissue_mm)+' mm^2', (0,420), cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 1, cv.LINE_AA)
cv.putText(data_HU_SoftTissue_Color,'The area of Torso is '+str(Area_Torso_mm)+' mm^2', (0,440), cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 1, cv.LINE_AA)
cv.putText(data_HU_SoftTissue_Color,'The centroid of Soft Tissue is '+str(cx_SoftTissue)+','+str(cy_SoftTissue), (0,460), cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 1, cv.LINE_AA)
cv.putText(data_HU_SoftTissue_Color,'The Dice Coefficient is '+str(DC), (0,480), cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 1, cv.LINE_AA)
cv.putText(data_HU_SoftTissue_Color,'The Area Percentatge is '+str(AreaPercentage), (0,500), cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 1, cv.LINE_AA)
cv.imshow("Soft tissue_C3_00606M (Target Region)",data_HU_SoftTissue_Color)
cv.waitKey(0)

# img_path = r'Desktop\Project_ZHJ\Results\SegResults-WithParameters\606'
# img_name = img_path + '\606_C3_SoftTissue.png'
cv.imwrite('636_'+VertebraName+'_SoftTissue.png', data_HU_SoftTissue_Color)
print('Successfully saved in the same folder')


# In[]: Missions

# Area Percentage done
# Centroid drawing for target one and evaluation one
# Show those on one pic and save
# Correct the T11 one
# Perimeter evaluation HOW TO???

# Remask to remove some small spots
# Re-evaluate and see the results change much or not

# Watershed algorithm for each slice -- results

# Vertebra detection















