# Automated-torso-soft-tissue-segmentation-software
This project comes from MSc project: Automated torso soft tissue segmentation for spine modelling in CT scan

--------------------Folder 'Codes': developed by Python----------------------------

SagittalBed&ConcaveArms.py: 
Sagittal Bed(CT scanner bed removal method 1) + Concave Arms(Arm removal method 1)

AxialBed&ConcaveArms.py:
Axial Bed(CT scanner bed removal method 2) + Concave Arms(Arm removal method 1)

SagittalBed&ManualWatershedArms.py: 
Sagittal Bed(CT scanner bed removal method 1) + Watershed Arms(Arm removal method 2)

AxialBed&ManualWatershedArms.py: 
Axial Bed(CT scanner bed removal method 2) + Watershed Arms(Arm removal method 2)

AutoSliceSelection.m
1. Aims: Return slice number for Python
2. User Guide:
    a. Input 'origin' -- 'The calculate value in MIMICS' from ResultsSummary.xls
    
-------------------Folder 'CT scan': 3 CT scan datasets-------------------------

MIMICS: orignal datasets
Dicom: dicom files exported from MIMICS
Markers: dicom files with the soft tissue mask(only target slices)

-------------------Folder 'Report'----------------------------------------------

Planning report
Final report
Final presentation slices
Midterm presentation slices: Include some experiment  result summaries

-------------------Folder 'Results'-----------------------------------------------

SegmentResults-Final
1. Red contours: automated soft tissue region
2. Yellow contours: manually masked soft tissue region
3. Parameters were written at the bottom
4. Dice coefficient was written at the bottom

ResultsSummary.xls
1. Centroids, vertebral body height, intervertebral disc height -- from Chloe
2. Areas, volumes, masses
3. Dice coefficient, area percentage
