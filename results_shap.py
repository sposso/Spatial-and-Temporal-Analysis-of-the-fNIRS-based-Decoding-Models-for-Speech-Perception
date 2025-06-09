#%%
import os
import numpy as np
from train_shap import machine_learning_exp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from audio_data import final_audio_data,final_all_audio_data
from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from deep_learning_utils import deep_learn
import shap


# Folder where the SHAP values will be stored 
out_folder = os.path.join(os.getcwd(), 'NEW_shap_audio_class')
os.makedirs(out_folder, exist_ok=True)

CONFIDENCE = 0.05  # stat confidence at 95 %
SEED= 22

subjects = [0,1,2,3,4,5,6,7]

dict_accuracies = {}
classes =2
# chromospheres in the blood : oxy-hemoglobin (HbO) and deocy-hemoglobin (HbR)
chroma = None
# t_min = onset of the event , t_max = end of the event (seconds)
t_min, t_max = 0.0, 18



for subject in subjects: 
    
    # Folder where the results for each subject will be stored 
    audio_folder = os.path.join(out_folder,f'_folder_subj_{subject}')
    os.makedirs(audio_folder, exist_ok=True)
    
    '''
     Data from the paper : The use of broad vs restricted regions of interest in functional near-infrared spectroscopy 
     for measuring cortical activation to auditory-only and visual-only speech"   
    '''
   
    DATASETS = {'audio_study': [t_min,t_max,chroma]}
    dataset = 'audio_study'

    # Table to store the results
    with open(f'{audio_folder}/summary_subj_{subject}.md', 'w') as w:
        w.write('# AUC table\n\n(Standard deviation on the cross-validation)')
        w.write('\n\n|Dataset|Chance level|')
        w.write('LDA (sd)|SVC (sd)|\n')
        w.write('|:---:|:---:|:---:|:---:|:---:|:---:|\n')
    with open(f'{audio_folder}/results_subj_{subject}.csv', 'w') as w:
        w.write('dataset;model;fold;AUC;hyperparameters\n')

    # Function to preprocess the data 
    #X shape : number of instances, number of channels, number of time points
    # Y (labels): number of instances
    X,Y,epochs = final_audio_data(subject,t_min,t_max,chroma)
    
    
    X_= X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    X = pd.DataFrame(X_)
    X.columns = X.columns.astype(str)
    
    #ch_names = np.arange(0, X.shape[1], 1)
    #ch_names = ch_names.astype(str).tolist()
    
    # Training and evaluation 
    shap_values_svc, hps_svc, svc,test_svc = machine_learning_exp("svc",X,Y)
    shap_values_lda, hps_lda, lda,test_lda = machine_learning_exp("lda", X, Y)
    

    # save the shap values for each folder created in the outer cross-vaidation
    np.savez(f'{audio_folder}/shap_values_svc_subject_{subject}', folder_1 = shap_values_svc[0],folder_2 = shap_values_svc[1],
                 folder_3 = shap_values_svc[2], folder_4 = shap_values_svc[3], folder_5= shap_values_svc[4])
        
    np.savez(f'{audio_folder}/shap_values_lda_subject_{subject}', folder_1 = shap_values_lda[0],folder_2 = shap_values_lda[1],
                 folder_3 = shap_values_lda[2], folder_4 = shap_values_lda[3], folder_5= shap_values_lda[4])
    
    results = {'LDA': [lda, hps_lda], 'SVC': [svc, hps_svc],
                }
    
    
    #################### Results Report ##########################################################################
    chance_level = np.around(1/classes, decimals=3)
    w_summary = open(f'{audio_folder}/summary_subj_{subject}.md', 'a')
    w_results = open(f'{audio_folder}/results_subj_{subject}.csv', 'a')
    w_summary.write(f'|{dataset}|{chance_level}|')
    for model in results.keys():
        w_summary.write(
            f'{np.around(np.mean(results[model][0]), decimals=3)} '
            f'({np.around(np.std(results[model][0]), decimals=3)})|')
        for fold, auc in enumerate(results[model][0]):
            hps = results[model][1][fold]
            w_results.write(f'{dataset};{model};{fold+1};{auc};"{hps}"\n')
    w_summary.write('\n')
    w_summary.close()
    w_results.close()
    dict_accuracies[dataset] = lda + svc 

    dict_accuracies['Model'] = list(np.repeat(list(results.keys()), len(lda)))
    df_accuracies = pd.DataFrame(dict_accuracies)
    df_accuracies = df_accuracies.melt(
        id_vars=['Model'], value_vars=list(DATASETS.keys()),
        var_name='Dataset', value_name='AUC')
    plt.figure(figsize=(16, 6))
    sns.barplot(data=df_accuracies, y='AUC', x='Dataset', hue='Model',
                capsize=.1, palette='colorblind')
    plt.axhline(1/2, color='blue', linestyle=':', label='2 classes chance level')
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc=6)
    plt.savefig(f'{audio_folder}/summary.png', bbox_inches='tight', dpi=1200)
    plt.close()
    
    print('Stats...')
    with open(f'{audio_folder}/stats.md', 'w') as w:
        df = pd.read_csv(f'{audio_folder}/results_subj_{subject}.csv', delimiter=';')
        w.write('## Comparison of model accuracies to chance level\n\n')
        w.write('|Dataset|Model|Shapiro p-value|Test|Statistic|p-value|\n')
        w.write('|:---:|:---:|:---:|:---:|:---:|:---:|\n')
        anova_table = ''
        for dataset in DATASETS.keys():
            dataset_accuracies = []
            chance_level = 1 / len(DATASETS[dataset])
            normality = True
            for model in results.keys():
                w.write(f'|{dataset}|{model}|')
                sub_df = df[(df['dataset'] == dataset) & (df['model'] == model)]
                accuracies = sub_df['AUC'].to_numpy()
                dataset_accuracies.append(accuracies)
                # Check normality of the distribution
                _, p_shap = stats.shapiro(accuracies)
                w.write(f'{p_shap}|')
                if p_shap > CONFIDENCE:
                    # t-test
                    s_tt, p_tt = stats.ttest_1samp(accuracies, chance_level,
                                                alternative='greater')
                    w.write(f't-test|{s_tt}|{p_tt}|\n')
                else:
                    normality = False
                    # Wilcoxon
                    s_wilcox, p_wilcox = stats.wilcoxon(accuracies-chance_level,
                                                        alternative='greater')
                    w.write(f'Wilcoxon|{s_wilcox}|{p_wilcox}|\n')
            _, p_bart = stats.bartlett(*dataset_accuracies)
            if normality and (p_bart > CONFIDENCE):
                s_anova, p_anova = stats.f_oneway(*dataset_accuracies)
                anova_table += f'|{dataset}|{p_bart}|ANOVA|{s_anova}|{p_anova}|\n'
            else:
                s_kru, p_kru = stats.kruskal(*dataset_accuracies)
                anova_table += f'|{dataset}|{p_bart}|Kruskal|{s_kru}|{p_kru}|\n'
        w.write('\n\n## Comparison of model accuracies to each other\n\n')
        w.write('|Dataset|Bartlett p-value|Test|Statistic|p-value|\n')
        w.write(f'|:---:|:---:|:---:|:---:|:---:|\n{anova_table}')



