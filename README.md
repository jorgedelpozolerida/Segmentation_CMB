# Segmentation_CMB
Repository containing results of my Research Project done in collaboration with [CEREBRIU](https://cerebriu.com/) company as part of my Msc in Data Science at [IT University of Copenhagen](https://en.itu.dk/). Please read full report in [Automated Segmentation of CMB](report/JorgedelPozoLerida_ResearchProject_AutomatedSegmentationofCMB.pdf)

## Abstract
Cerebral Microbleeds (CMBs) are crucial neuroimaging biomarkers associated with medical conditions such as stroke, intracranial hemorrhage, and cerebral small vessel disease. They are detectable as hypointensities on magnetic resonance images (MRI) in T2*-weighted or susceptibility-weighted sequences. 

Identifying CMBs is a time-consuming and error-prone task for radiologists, making the need for automatic detection critical. Yet, it remains a challenging endeavor due to the small size and quantity of CMBs, scarcity of publicly available annotated data, and their resemblance to various other mimics among other things. This complexity hinders the development of a clinically integrated automated solution. 

In response to these challenges, this study carefully reviewed the literature on this topic and tested a commonly used architecture, U-Net, for the segmentation and detection of CMBs using the public VALDO dataset. Adhering to the latest research guidelines, the study achieved a recall of 0.71, a precision of 0.44, and an F1 score of 0.54, with an average of 1.5 and 0.9 false positives per subject and per CMB respectively. Concurrently, a new clinically relevant dataset specifically tailored for CMB segmentation was developed, to be utilized in future work.

## Contributions
The project's main contributions can be summarized into:
* Generating a new, clinically relevant dataset for CMB
segmentation to be used in future work. See
* Testing U-Net architecture for CMB segmentation on the VALDO dataset, following latest literature guidance.


### Created dataset for CMB segmentation
A total of 70 cases were selected containing diverse pathologies and coming from different locations. Scanners parameters are also diverse. The annotation protocol and framework as designed as part of this work. Weak 2D anotations were turned into 3D masks for the CMBs 

![](img/table1.png)
![](img/table2.png)


### Segmentation of CMBs with 3D U-Net 

A 3D-Unet was trained in different setups to learn the task of segmenting and detecting CMBs, achieving satisfactory results given the complexity of the task and the state-of-the-art. As with many other approaches in literature, FPs were the main problem. 
![](img/table5.png)
![](img/figure8.png)


## Repository overview
The repository contains the following folders.

