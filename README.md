# Automated-torso-soft-tissue-segmentation-software
This project comes from MSc project: Automated torso soft tissue segmentation for spine modelling in CT scan

------------------------------------Folder 'Codes': developed by Python---------------------------------------------------

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
