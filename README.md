# Spatial and Temporal Analysis of the fNIRS-based Decoding Models for Speech Perception using SHAP

## Abtsract 

Speech decoding extracts speech information from neural activity. Previous studies have demonstrated that functional near-infrared spectroscopy (fNIRS) contains information suitable for language decoding. However, the interpretability of speech decoders remains unexplored. This study aims to identify the relevant brain regions and time points that enable neural decoders to distinguish between speech perception and silence. We trained a linear support vector classifier (SVC) and a linear discriminant analysis (LDA), and conducted post-hoc analysis using the Shapley additive explanations (SHAP) technique to identify the spatiotemporal patterns contributing to the neural decoder’s performance. We utilized a public fNIRS dataset, which comprises recordings from eight adults during the auditory perception of speech and silence. Our results indicate that features from oxyhemoglobin (HbO) and deoxyhemoglobin (HbR) are relevant and that the inferior frontal gyrus (IFG) and Wernicke’s area are key for differentiating speech perception from silence, which aligns with established neurophysiological processes. Temporally, we observed a subtle increase in feature relevance at the stimulus onset (t=0 s) and around 6-10 s across subjects, which may be related to the initial dip and the peak of the hemodynamic response. These results suggest that studies could use key spatiotemporal fNIRS features to improve speech decoding performance.

## DATA

The data used is publicly available from the study ***"The use of broad vs restricted regions of interest in functional near-infrared spectroscopy for measuring cortical activation to auditory-only and visual-only speech"***. For more details about the dataset and the preprocessing steps, please refer to this [description](https://github.com/sposso/fNIRS-preprocessing-guide)



## Explanation of the repository's scripts

1. **train_shap.py** :  Contains functions to train Linear SVC and LDA models, and compute SHAP values for the test set samples.

   <ins>Key points to consider</ins> :
   
    - To address class imbalance (18 trials for the auditory-only condition and 10 for the resting condition), we applied the Adaptive Synthetic (ADASYN) sampling method
    - Following the ML methodology outlined [here](https://doi.org/10.3389/fnrgo.2023.994969)  for correctly evaluating models applied to fNIRS data, we tested regularization parameter          values for the SVC: 0.001, 0.01, 0.1, and 1, with a maximum of 250,000 iterations to ensure convergence
    - We employed a nested cross-validation to optimize and evaluate the models without bias. The outer cross-validation consisted of 5-fold cross-validation to define the test set.             Meanwhile, the inner cross-validation, designed for hyperparameter tuning, used a 3-fold cross-validation to separate the training and validation sets.
   - Use of **SHAP technique**: After the SVC and LDA models were trained to distinguish between the silent condition and speech perception, we performed the SHAP-based analysis. The 
     SHAP explainer was applied to all training samples, and SHAP values were computed for all test samples in each fold. Since this is a binary classification task, SHAP decomposes each 
     prediction score into contributions toward the positive and negative classes. For each sample, the SHAP values indicate how individual features contribute to the prediction; 
     positive SHAP values for samples from the positive class (silence condition)  and negative SHAP values for samples of the negative class (auditory condition) help predictions. To 
     assess the global importance of each feature, we multiply each instance's SHAP values by its corresponding label ${-1,1}$; in this way, we ensure that positive values per instance 
     mean the feature contributes to the prediction.
   
3. **results_shap.py** : Loads the data and calls the training function from train_shap.py to train the models. It also saves the performance of the models and their respective SHAP values.  It also performs statistical tests to assess: (1) whether each model performs significantly above chance level (p < 0.05) and (2) whether there are performance differences between models using one-way ANOVA. 

   <ins>Key points to consider</ins>:
   - The SHAP VALUES are saved in an .npz file, one per model. 
   - Each .npz file contains 5 sets of SHAP values, corresponding to the 5 outer folds used in cross-validation (i.e., five models are trained and evaluated independently). 
   
5. **graphs.py** : Analyzes the relevance of spatiotemporal features to model performance using SHAP values.

    <ins>Key points to consider</ins>:

    - Loads the .npz files containing SHAP values saved during training. Each SHAP array has shape (num_instances × num_features), where num_features = channels × time points.

    - SHAP values are averaged across instances, reshaped to (channels × time points), and normalized to the [0, 1] range.

    - Coefficients are then summed separately along the spatial (channels) and temporal (time points) dimensions for each subject across all outer cross-validation folds.

   -  Since the number of channels can vary across subjects due to preprocessing (e.g., discarding optodes with poor scalp connectivity), spatial analyses are conducted at the brain 
      region level rather than the individual channel level.

    - Features derived from HbO and HbR signals are analyzed separately to provide a more detailed assessment of their contributions to decoding performance.


## Results

### Decoding Performances

| Method | sub.0      | sub.1      | sub.2      | sub.3      | sub.4      | sub.5      | sub.6      | sub.7      |
|--------|------------|------------|------------|------------|------------|------------|------------|------------|
| **SVC** | 0.76 ± 0.17 | 0.90 ± 0.12 | 0.63 ± 0.16 | 0.63 ± 0.16 | 0.98 ± 0.05 | 0.72 ± 0.07 | 0.85 ± 0.20 | 0.63 ± 0.14 |
| **LDA** | 0.79 ± 0.17 | 0.84 ± 0.16 | 0.63 ± 0.11 | 0.58 ± 0.07 | 0.94 ± 0.07 | 0.64 ± 0.10 | 0.73 ± 0.16 | 0.67 ± 0.16 |

### Subject-wise relevant spatiotemporal features identified by SHAP for Linear SVC and LDA

#### Dictionary to understand the brain regions referred to in the graphs:

**<ins>Au_A_R and Au_A_L</ins>** = Right and left auditory A region that  is located in the rostral aspect of the superior temporal gyrus (also referred to as Heschl’s gyrus). This region is responsible for transforming basic acoustic features into more complex variations. 

**<ins>Au_B_R and Au_B_L</ins>** = Right and left auditory B region that  is located in the caudal aspect of the superior temporal gyrus ( Wernicke’s area). This area is involved in phonological analysis and supports working memory processes during language tasks 

**<ins>IFG</ins>** = Left inferior frontal gyrus, which houses Broca’s area, and is involved in phonological analysis and supports working memory processes during language tasks.

**<ins>V_A</ins>** = Visual A region that includes Cuneus and superior occipital gyrus.

**<ins>V_B</ins>** = Visual B region that includes  middle occipital gyrus. 


### Relevant brain areas for all subjects according to the SHAP technique 

##### SVM 

![Results of the SVC](https://github.com/sposso/Spatial-and-Temporal-Analysis-of-the-fNIRS-based-Decoding-Models-for-Speech-Perception/blob/main/Figures/brain_area_counts_grid_shap.png)

##### LDA 

![Results of the LDA](https://github.com/sposso/Spatial-and-Temporal-Analysis-of-the-fNIRS-based-Decoding-Models-for-Speech-Perception/blob/main/Figures/brain_area_counts_grid_lda_shap.png)

#### Relevant time points for all subjects according to the SHAP technique 

##### SVM 

![Results of the SVC](https://github.com/sposso/Spatial-and-Temporal-Analysis-of-the-fNIRS-based-Decoding-Models-for-Speech-Perception/blob/main/Figures/count_times_grid_shap.png)


#### LDA

![Results of the LDA](https://github.com/sposso/Spatial-and-Temporal-Analysis-of-the-fNIRS-based-Decoding-Models-for-Speech-Perception/blob/main/Figures/count_times_grid_lda_shap.png)













