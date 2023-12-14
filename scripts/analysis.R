################################################################################
#' Title: 
#' Author: Jorge del Pozo Lerida
#' Date: 2023-11-30
#' Description: 
################################################################################

## Setup -------------------------------------------------------------------

# Load necessary packages
library(tidyverse)

data <- read_csv("/home/cerebriu/data/RESEARCH/Segmentation_CMB/data/CMB_processed.csv")


d <- data %>% 
  distinct(subject_id, .keep_all = T)

mean(d$n_CMB)
j<-d  %>% 
  filter(n_CMB <60) 
hist(j$n_CMB)

j<-data  
  max(j$n_voxels)
hist(j$n_voxels)

d <- read_csv("/home/cerebriu/data/RESEARCH/Segmentation_CMB/data/VALDO_processed_v1_metadata.csv")  
d2 <- read_csv("/home/cerebriu/data/RESEARCH/Segmentation_CMB/data/VALDO_processed_v2_metadata.csv")  



max(d$X_dim)
max(d$Y_dim)
  