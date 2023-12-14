################################################################################
#' Title: 
#' Author: Jorge del Pozo Lerida
#' Date: 2023-12-10
#' Description: 
################################################################################

## Setup -------------------------------------------------------------------

# Load necessary packages
source("src/library_imports.R")

# Load custom functions
source("src/functions_utils.R")



CMB_prio <- read_csv("/home/cerebriu/data/RESEARCH/Segmentation_CMB/data/CMB_detected_SilviaAnnotation.csv")
CMB_meddare <- read_csv("/home/cerebriu/data/RESEARCH/Segmentation_CMB/data/CMB_MedDARE_data.csv") %>% 
  distinct(StudyInstanceUID, .keep_all = T)
sequence_metadata <- read_csv("/home/cerebriu/data/DM/MyCerebriu/Pathology_Overview/all_sequences_final.csv")
study_metadata <- read_csv("/home/cerebriu/data/DM/MyCerebriu/Pathology_Overview/all_studies_final.csv")




# Match all data ----------------------------------------------------------
data_all <- sequence_metadata %>% 
  filter(StudyInstanceUID %in% CMB_meddare$StudyInstanceUID) %>% 
  # identify only SWI or T2S
  filter(grepl("T2S|SWI", CRBSeriesDescription, ignore.case = TRUE)) %>%
  group_by(StudyInstanceUID) %>% 
  mutate(n=n()) %>% 
  filter(n==1) %>% 

  # Add study-level
  left_join(study_metadata %>% 
              filter(StudyInstanceUID %in% CMB_meddare$StudyInstanceUID) %>% 
              select(-Dataset, -Step),
            by="StudyInstanceUID") %>% 
  # Add desired columns
  mutate(
    Resolution= paste0("(", Rows, ", " ,Columns, ", ", Slices,  ")"),
    country= sapply(str_split(Dataset, "-"), `[`, 1),
    country = case_when(
      country == 'BR' ~ "Brazil",
      country == "IN" ~ "India",
      country == "US" ~ "U.S.A"
    ),
    MagneticFieldStrength = round(as.numeric(MagneticFieldStrength), 2),
    MagneticFieldStrength = case_when(
      MagneticFieldStrength== "15000" ~ 1.5,
      TRUE ~ MagneticFieldStrength
    ),
    Demographics = "Not available",
    Location = country,
    RepetitionTime = round(as.numeric(RepetitionTime)),
    EchoTime = round(as.numeric(EchoTime)),
    `TR/TE (ms)`= paste0(RepetitionTime, "/", EchoTime),
    `TR (ms)`= RepetitionTime,
    `TE (ms)`= EchoTime,
    
    `Scanner Type` = paste0(Manufacturer, " ", MagneticFieldStrength, "T" ),
    `Scanner Model` = ManufacturerModelName ,
    `Flip Angle` = FlipAngle,
    voxel_vals_1 = round(as.numeric(sapply(str_extract_all(PixelSpacing, "\\d+\\.?\\d*"), `[[`, 1)), 2),
    voxel_vals_2 = round(as.numeric(sapply(str_extract_all(PixelSpacing, "\\d+\\.?\\d*"), `[[`, 2)),2),
    `Voxel Size (mm3)` = paste0(voxel_vals_1, "x",voxel_vals_2, "x", SliceThickness),
    `Seq. Type`= CRBSeriesDescription,
    Dataset_big = sapply(str_split(Dataset, "-"), `[`, 2),
    Hospital= case_when(
      Dataset_big == "FIDI" ~ "Source 1",
      Dataset_big == "BodyScanData"~ "Source 2",
      Dataset_big == "Victoria"~ "Source 3",
      Dataset_big == "Aarthi" ~ "Source 4",
      Dataset_big == "SUNY"~ "Source 5"
    )
  ) %>% 
  # Select relevant
  select(StudyInstanceUID, Demographics, Hospital, Location, `Scanner Type`, `Scanner Model`, `Seq. Type`, `TR/TE (ms)`,`TR (ms)`, `TE (ms)`, `Flip Angle`, Resolution, `Voxel Size (mm3)`)





# Build table -------------------------------------------------------------

# Function to calculate percentages and format output
calc_percent <- function(x) {
  freq <- table(x)
  percent <- round(100 * freq / sum(freq), 2)
  
  # Check if only one category exists
  if (length(freq) == 1) {
    return(names(freq))
  } else {
    return(paste(paste0(round(percent), "%"), names(freq), sep=": ", collapse=", "))
  }
}

# Aggregating data
summary_table <- data_all %>%
  group_by(Hospital) %>%
  summarise(
    # Demographics = calc_percent(Demographics),
    Location = calc_percent(Location),
    `Scanner Type` = calc_percent(`Scanner Type`),
    `Scanner Model` = calc_percent(`Scanner Model`),
    `Seq. Type` = calc_percent(`Seq. Type`),
    `TR/TE (ms)` = calc_percent(`TR/TE (ms)`),
    # `TR (ms)` = calc_percent(`TR (ms)`),
    # `TE (ms)` = calc_percent(`TE (ms)`),
    `Flip Angle` = calc_percent(`Flip Angle`),
    Resolution = calc_percent(Resolution),
    `Voxel Size (mm3)` = calc_percent(`Voxel Size (mm3)`),
    `# patients` = n()
  ) %>%
  ungroup()

summary_table
write_csv(summary_table, "/home/cerebriu/data/RESEARCH/Segmentation_CMB/data/summary_newdataset_scannerparams.csv")



# Pathologies -------------------------------------------------------------

# Convert CRB_ columns to boolean
df <- CMB_meddare %>%
  mutate(
    CRB_Infarct = as.logical(CRB_Infarct),
    CRB_Hemorrhage = as.logical(CRB_Hemorrhage),
    CRB_Tumor = as.logical(CRB_Tumor)
  )

df2 <- df %>%
  select(-contains("location")) %>%
  pivot_longer(cols = c(starts_with("other"), "infarct", "hemorrhage", "tumor"), names_to = "Pathology_Type") %>%
  group_by(StudyInstanceUID, Pathology_Type, value) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = value, values_from = count, values_fill = 0) %>%
  ungroup() %>%
  select(-Pathology_Type)


result <- df %>%
  summarize(
    CRB_Infarct_count = sum(CRB_Infarct),
    CRB_Hemorrhage_count = sum(CRB_Hemorrhage),
    CRB_Tumor_count = sum(CRB_Tumor)
  ) %>%
  ungroup()



#subtypes
# Get the list of column names in df2 excluding StudyInstanceUID
column_names <- setdiff(colnames(df2), c("StudyInstanceUID", "Not present"))

# Initialize an empty list to store the results
column_counts <- list()

# Loop over the column names and calculate counts
for (col_name in column_names) {
  counts <- df2 %>%
    summarise(count = sum(!!sym(col_name))) %>%
    rename(!!paste0(col_name, "_count") := count)
  
  # Append the counts to the list
  column_counts[[col_name]] <- counts
}

# Combine the counts into a single dataframe
result_counts <- bind_cols(column_counts)

# Print the counts for each column
print(result_counts)




unique_values_1 <- unique(df$additional_findings_1)
unique_values_2 <- unique(df$additional_findings_2)
unique_values_3 <- unique(df$additional_findings_3)

unique_vals <- unique(
  c(unique_values_1, 
  unique_values_2,
  unique_values_3)
)




