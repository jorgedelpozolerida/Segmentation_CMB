################################################################################
#' Title: 
#' Author: Jorge del Pozo Lerida
#' Date: 2023-10-11
#' Description: 
################################################################################

## Setup -------------------------------------------------------------------

# Load necessary packages
source("src/library_imports.R")

# Load custom functions
source("src/functions_utils.R")

phases <- c(6, 7, 8, 9, 10, 11)

# Create an empty dataframe to store the results
results <- data.frame(id = character(),
                      phase = numeric(),
                      column_name = character(),
                      value = character(),
                      stringsAsFactors = FALSE)

# Patterns to search for
patterns <- c(
  "microbleeds", "micro-bleeds", "bleeds", "microhemorrhages", "micro-hemorrhages"
              # ,"microhemorrhages (<10 mm)"
              # ,"microhemorrhages (<10 mm)"
              )

# Data loading
data_src_phases <- list()
all_data <- data.frame()
for (phase_n in phases) {
  data_src_path <- paste0("/home/cerebriu/Downloads/Phase", phase_n, "_merged.xlsx") 
  temp <- readxl::read_excel(data_src_path, na="NA") %>% 
    mutate_at(c("CRB_Infarct", "CRB_Tumor", "CRB_Hemorrhage"),
              ~case_when(
                . == "TRUE" | . == "yes" ~ TRUE,
                TRUE ~ FALSE
              )) %>% 
    mutate(CRB_include=tolower(CRB_include))
  
  data_src_phases[[paste0("phase", phase_n)]] <- temp
  all_data <- bind_rows(all_data, temp)
}



for (phase in phases){
  data_src_path <- paste0("/home/cerebriu/Downloads/Phase", phase, "_merged.xlsx")
  data_src <- readxl::read_excel(data_src_path, na="NA")
  
  # Convert data to lowercase
  data_src <- data_src %>%
    mutate(across(everything(), tolower, .names = "lc_{.col}"))
  
  for (col in colnames(data_src)) {
    if (col != "id") {
      for (pattern in patterns) {
        matched_rows <- which(str_detect(data_src[[col]], pattern))
        
        if (length(matched_rows) > 0) {
          results <- rbind(results, data.frame(id = data_src$id[matched_rows],
                                               phase = phase,
                                               column_name = col,
                                               value = data_src[[col]][matched_rows]))
        }
      }
    }
  }
}

results  <- results %>% 
  mutate(StudyInstanceUID = sapply(str_split(id, "_"), `[`, 1)) %>% 
  group_by(StudyInstanceUID) %>% 
  mutate(n=n()) %>% 
  ungroup() %>% 
  relocate(id, StudyInstanceUID) %>% 
  filter(str_detect(column_name, "additional_findings"))
  # filter(!(column_name %in% c("Report", "lc_Report", "ParsedImpressions", "lc_ParsedImpressions")))

results %>% distinct(StudyInstanceUID) %>% nrow()
results %>% filter(value == "microhemorrhages (<10 mm)") %>% distinct(StudyInstanceUID) %>% nrow()

results_distinct <- results %>% distinct(StudyInstanceUID, .keep_all = T)

data_out_final <- results_distinct %>% 
  select(-StudyInstanceUID) %>% 
  left_join(all_data, by=c("id")) %>% 
  select(id, StudyInstanceUID, contains("CRB"))

# Prioritize --------------------------------------------------------------
range01 <- function(x, e=0.00001){
  (x-min(x))/(max(x)-min(x))
  
  }

CMB_prio <- results_distinct %>% 
  select(id, StudyInstanceUID, column_name, value) %>% 
  mutate(Phase = sapply(str_split(id, "_"), `[`, 3)) %>%
  select(-id) %>% 
  mutate(Phase=ifelse(Phase =="pilot", "phase3", Phase)) %>% 
  rename(col_detected=column_name) %>% 
  left_join(data_out_final %>% select(StudyInstanceUID, contains("CRB")),
            by="StudyInstanceUID") %>% 
  select(-CRB_Other) %>% 
  filter(CRB_quality == "sufficient") %>% 
  mutate(
    is_good = ifelse(value == "microhemorrhages (<10 mm)", TRUE, FALSE),
    is_perfect = ifelse(
      CRB_Infarct == FALSE  &    CRB_Hemorrhage == FALSE &   CRB_Tumor == F,
      T, F),
    CRB_include = ifelse(CRB_include == "yes", TRUE, FALSE)
  ) %>%
  arrange(
    desc(CRB_include),
    desc(is_good),
    desc(is_perfect),
    desc(CRB_Tumor)
    
  ) %>%
  mutate(priority = row_number()) %>% 
  mutate(
    user_mail="si@cerebriu.com"
  ) %>% 
  mutate(priority = range01(rev(priority))
           
           ) %>% 
  left_join( all_data %>% 
               distinct(StudyInstanceUID, Dataset),
             by="StudyInstanceUID")

summ_CRB <- CMB_prio %>% 
  group_by(CRB_Infarct,CRB_Tumor, CRB_Hemorrhage) %>% 
  summarise(n=n())



write_csv(CMB_prio, "/home/cerebriu/data/DM/MyCerebriu/CMB/CMB_detected.csv")
write_csv(CMB_prio %>%
            mutate(id = row_number()) %>% 
            select(id, contains("CRB"), priority)
            , "/home/cerebriu/data/RESEARCH/Segmentation_CMB/data/CMB_detected_SilviaAnnotation.csv")

# With all meddare data
all_data_results <- all_data %>% 
  filter(StudyInstanceUID %in% CMB_prio$StudyInstanceUID) %>% 
  left_join(CMB_prio %>% select(StudyInstanceUID, priority)) %>% 
  relocate(id, StudyInstanceUID, priority)
write_csv(all_data_results, "/home/cerebriu/data/DM/MyCerebriu/CMB/CMB_MedDARE_data.csv")

