
# Jorge's Weekly Meeting Notes

* [9 October 2023](#date-9-october-2023)
* [30 October 2023](#date-30-october-2023)
* [10 November 2023](#date-10-november-2023)
<!-- * [13 October 2023](#date-13-october-2023)
* [16 October 2023](#date-16-october-2023)
* [20 October 2023](#date-20-october-2023)
* [23 October 2023](#date-23-october-2023)
* [3 November 2023](#date-3-november-2023)
* [10 November 2023](#date-10-november-2023)
* [13 November 2023](#date-13-november-2023)
* [17 November 2023](#date-17-november-2023)
* [20 November 2023](#date-20-november-2023)
* [24 November 2023](#date-24-november-2023)
* [27 November 2023](#date-27-november-2023)
* [1 December 2023](#date-1-december-2023)
* [4 December 2023](#date-4-december-2023)
* [8 December 2023](#date-8-december-2023)
* [11 December 2023](#date-11-december-2023)
* [15 December 2023](#date-15-december-2023) -->
* [Template](#date-template)

<br><br><br><br><br>
<br><br><br><br><br>



### Date: Template

#### Who did you help this week?

Replace this text with a one/two sentence description of who you helped this week and how.


#### What helped you this week?

Replace this text with a one/two sentence description of what helped you this week and how.

#### What did you achieve/do?

* Replace this text with a bullet point list of what you achieved this week.
* It's ok if your list is only one bullet point long!

#### What did you struggle with?

* Replace this text with a bullet point list of where you struggled this week.
* It's ok if your list is only one bullet point long!

#### What would you like to work on next ?

* Replace this text with a bullet point list of what you would like to work on next week.
* It's ok if your list is only one bullet point long!
* Try to estimate how long each task will take.

#### Where do you need help from Veronika?

* Replace this text with a bullet point list of what you need help from Veronica on.
* It's ok if your list is only one bullet point long!
* Try to estimate how long each task will take.

#### Others


This space is yours to add to as needed.

<br><br><br><br><br>


### Date: 9 October 2023


#### What did you achieve/do?
* familiarized with ClearML (MLOps infrastructure) and how to code in it: performed some basic experiment with brats19 dataset
* analyzed VALDO dataset images. Found some issues and a lot of preprocessing needed. See metadata: [VALDO_metadata.csv](../data/VALDO_metadata.csv) 
* Created structure for saving info from literature review into a DB in Notion

* Collected >20 articles to read - mostly review articles - to get big picture of Segmentation task in general (following funnel). For now just collected a few about CMB segmentation specifically.
* Found another public dataset with CMB: [new dataset](https://appsrv.cse.cuhk.edu.hk/~qdou/cmb-3dcnn/cmb-3dcnn.html)

#### What did you struggle with?
* Understand what preprocessing is needed for VALDO dataset
* Dataset not as clean as expected: 
    * There are 12 label maps for CMB that are empty (all background) - out of 72 subjects
    * A third of the annotated CMB consist of  <10 pixels in size
    * For the three available MR sequences - T1, T2, T2s - I found a total of 60 out of 216 available sequences having many "nan" values
    * Images having diverse sizes, voxel dims and intensity ranges

#### What would you like to work on next ?
1. Finish preprocessing of dataset
2. Implement unet and see results
3. Perform more in depth literature review of CMB specifically

#### Where do you need help from Veronika?
* Talk about desired points to discuss for Tuesday 10th meeting with CEREBRIU (tomorrow)
* Can one have free access to articles through ITU? (SciHub did not help in some cases / books)
* Is it possible agree on some date ranges for Research Project presentation in January? (planning to visit parents in Spain, ticket prices skyrocketting)
* Not sure what page limit is for RP report... guidelines seem to be for MSc thesis report. What is it normally?

#### Others

* Apparently there is no internal pixel-level annotated data for CMB in my company, only image-level... not sure how this could help
* I have contacted the main author of VALDO challenge to ask for extra information about dataset preprocessing and issues found, waiting for response - also requested the test data (not provided in challenge and twice as abundant)
* I thought of following this funnel strategy for the literature review:
    - First look at review papers for segmentation task in general
    - Then review papers for MEDICAL imaging segmentation
    - Then review papers for medical imaging segmentation with DL/automated methods
    - Then research/review papers for segmentation of CMB specifically
* Future TODOs:
    - Narrow down better to literature search to CMB segmentation only, filling a table with different approaches so far and metadata about them
    - Given low number of labeled data:
        - Implement basic data augmentations
        - Given small size of target, implement adapted patch sampling strategy 

  

<br><br><br><br><br>

### Date: 13 October 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* Agree on points to bring up for tomorrow's meeting with Mathias&Silvia (CEREBRIU)

#### Others

* n/a

<br><br><br><br><br>

### Date: 16 October 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 20 October 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 23 October 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 30 October 2023





#### What did you achieve/do?
* Coordinate annotation of microbleeds for MSc thesis (IN PROGRESS):
    - Identify from internal data which have potentially microbleeds: ~70
    - Start planning how to set up annotation environment on Redbrick platform for Silvia Ingala to annotate

* Preprocessing of VALDO dataset (DONE, file: [data_preprocessing.py](../scripts/data_preprocessing.py))
    1. Perform QC while loading data (handle Nans in MRI data, clean masks)
    2. Resample and Standardize (T2s to isotropic, the rest resampled to T2s)
    3. Crop (using brain mask)
    4. Concatenate (stack together into single file)

* Run U-Net model on preprocessed dataset using a "whole-mri" strategy (downsampling to small 64x64x64 size and inputtig to u-net). Results quite bad, see: [unnet_exp_1.png](../img/unnet_exp_1.png)

* Split function for train-val-test maintaining proportion of healthy/unhealthy. See [generate_split.py](../scripts/generate_split.py)

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
1. Literature research on methods for CMB segmentation
2. Implement U-Net but with nicer patch sampling startegy + other common tricks to see results.

#### Where do you need help from Veronika?
* Is there a standarized way of preprocessing data before feeding to CNN? Specially with regard to order of steps applied...
* Best metric to evaluate when class imbalance is so big? CMB are tiny

#### Others

* Talked with SAP about dates for presentation, they ensured presnetation would take place before 19th January.
* For the microbleeds to be annotated for my MSc thesis, I will have:
    - sequence and study level metadata about: 
        - presence/absence of various tumor/hemorrhages/infarcts
        - image quality
    - radiological reports
    - scanner parameters
* Of the ~70 potential microbleeds cases, these contains very likely some other kind of pathology
(Tumor, Hemorrhage, Infarct; of various types)

<br><br><br><br><br>


### Date: 3 November 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 10 November 2023



#### What did you achieve/do?
* Changes in preprocessing:
    - voxel size of 0.5 for isotropic (most of the images have half mm size, pitty to lose that)(not sure which size I'll use yet)

* Implemented my own region growing algorithm to generate CMB microbleed mask from seeds
    - Algorithm starts at seeds and adds iteratively neighbours based on tolerance. Tolerance level set for every case individually.
    - Tested on VALDO data to see how it looks:
        - Visually it looks good
        - When looking at Dice scores (against GT) it varies a lot from sample to sample
        - The smaller the CMb the worse the performance
    - Results can be seen in [region_growing.ipynb](../scripts/region_growing.ipynb)

* Configured credentials to use a different server for running experiments (experiencing tensorflow-Python-CUDA versions trouble)
* Finished setting up annotation framework for CMB in-house:
    - Had alignment meetings with silvia to define annotation process:
        - Agreed on some taxonomy to use, see [taxonomy_CMB.json](../data/taxonomy_CMB.json) 
        - Guidelines based partially on: [bombs-userguide.pdf](../papers/CMB/bombs-userguide.pdf)
    - An image of how one annotation looks can be found here: [annotation_example.png](../scripts/annotation_example.png)
    - There are 43 cases with T2S and 22 cases with SWI.
    - I created a specific order of annotation, prioritizing cases with no patologies and then tumors and then the rest (and other things)


#### What did you struggle with?
* Implementing my own region growing, libraries like sitk did not seem to work fine...
* Technical issues, not able to use GPUs from company server with tensorflow... not sure what solution is
(big problem actually)
* Spent a lot of time coordinating, analyzing data and scripting to generate the whole annottation setup...

#### What would you like to work on next ?
* Literature review of methods used for CMB!! Create a nice table summarizing what people do with what and how
* More implementation:
    - Use deeper U-net
    - Add transforms ot images before loading into model: normalization 
    - Add augmentations (e.g. intensity augmentations) + normalization again
    - Implement patch sampling strategy...

#### Where do you need help from Veronika?
* Do you know a nice implementation of region growing algorithm? Mine is definitly not perfect and not even sure if acceptable, although not bad either..
* When collecting methods in literature for CMB segmentation, would you reocmmend looking at some specific points  in special? I'm thinking of creating a nice sumamry table but need to be speicfic on what to look for...
* I am spending a lot of time in curating and generating a dataset of CMB. I guess this was not expected as part of the project proposal, but I'm thiking it's something equally valid to include in report along with the rest. What do you think?


#### Others

* I think my generated region growing masks sometimes look better than interpolated, perhaps worth it to reprocess interpolated ones 
* Nice overview of CMBs available in VALDO dataset here: [CMB_processed.csv](../data/CMB_processed.csv)
* Nice overview of CMBs to be annoatted by Silvia here: [CMB_detected_SilviaAnnotation.csv](../data/CMB_detected_SilviaAnnotation.csv)


<br><br><br><br><br>

### Date: 13 November 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 17 November 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 20 November 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>



### Date: 24 November 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 27 November 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 1 December 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 4 December 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 8 December 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 11 December 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>

### Date: 15 December 2023





#### What did you achieve/do?
* n/a

#### What did you struggle with?
* n/a

#### What would you like to work on next ?
* n/a

#### Where do you need help from Veronika?
* n/a

#### Others

* n/a

<br><br><br><br><br>



