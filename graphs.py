#%%
import numpy as np 
import os 
from audio_data import final_audio_data
import matplotlib.pyplot as plt

import torch 
from torch.nn import AdaptiveAvgPool2d, Conv2d
import pandas as pd

#Folder where the SHAP values are stored. 
out_folder = os.path.join(os.getcwd(), 'NEW_shap')

# Folder where the figures generated in this script will be saved
folder_ = os.path.join(os.getcwd(), 'NEW_shape_figures')
os.makedirs(folder_, exist_ok=True)

# This mapping will be used to change the name of the fNIRS channels 
translation = str.maketrans("", "", "DShbo ")

list_dataframes = []
list_dataframes_hbr = []

dataframes_per_subjects = []
dataframes_per_subjects_hbr = []

# Channels that cover different brain regions
# Name explanation: 1_1 means emisor 1 is connected to receptor 1, and this connection create a fnirs channels. 
# This information was found in the study's supplementary information  that relase the dataset.
channels_to_brain_areas ={"IFG":["1_1","2_1","3_1","3_2","4_1","4_2","5_1","5_2"],
                          "Au_A_L":["6_3","6_4","8_3","8_4"],
                          "Au_A_R":["10_8","10_9","11_8","11_9"],
                          "Au_B_L":["7_5","7_6","8_5","8_6","8_7","9_5","9_6","9_7"],
                          "Au_B_R":["11_10","11_11","11_12","12_10","12_11","13_10","13_11","13_12"],
                          "V_A":["14_13","14_14","14_15"],
                          "V_B":["15_13","15_14","16_14","16_15"]}

all_subjects_count_times = np.zeros((8,18))
all_subjects_count_times_hbr = np.zeros((8,18))

#number of subjects
subjects = [0,1,2,3,4,5,6,7]

count_subjects = 0


for subject in subjects: 
    

    audio_folder = os.path.join(out_folder,f'_folder_subj_{subject}')
    
    #loading shap  balues from the SVC model
    weights_path= f'shap_values_svc_subject_{subject}.npz'
    
    #Loading  shap values from the LDA model
    #weights_path= f'shap_values_lda_subject_{subject}.npz'
    

    weight_path = os.path.join(audio_folder, weights_path)
    
    weights = np.load(weight_path)

    # End of the event (18s)
    t_max = 18

    X,_,epochs = final_audio_data(subject,0.0,t_max,None)
    ch_names = epochs.ch_names
    
    # renaming  fnirs channels 
    
    hbo_names_o = [name.translate(translation)for name in ch_names if 'hbo' in name]
    

    hbo_names = [name for name in hbo_names_o[::-1]]
    
    #print(hbo_names_o)

    count_times = np.zeros((5,18))
    count_times_hbr = np.zeros((5,18))
    count_channels = np.zeros((5,X.shape[1]//2))
    count_channels_hbr = np.zeros((5,X.shape[1]//2))

    count =0

    for folder in weights.files:
            
        folder_weights = weights[folder]
        
        # Mean of the SHAP  values across all instances  from one test folder 
        folder_weights = np.mean(folder_weights, axis=0)
        
        # Reshape to (n_channels, n_times)
        folder_weights = folder_weights.reshape(X.shape[1],X.shape[2])
        
        
        #Normalize weights
        folder_weights = (folder_weights - np.min(folder_weights)) / (np.max(folder_weights) - np.min(folder_weights))
        
        # Binary weights and folder weights are equal. 
        # I changed the name because I was working on something different before, but it does not make any difference now 
        binary_weights = folder_weights

        # Split hbo and hbr to make a separate analysis. 
        
        
        binary_hbo_weights = binary_weights[:X.shape[1]//2,:]
        binary_hbr_weights = binary_weights[X.shape[1]//2:,:]
        
        
        #hbo weights to tensor
        
        tensor_binary_hbo_weights = torch.tensor(binary_hbo_weights, dtype=torch.float32)
        #Reshape to N, C, H, W
        tensor_binary_hbo_weights = tensor_binary_hbo_weights.unsqueeze(0).unsqueeze(0)

        tensor_binary_hbr_weights = torch.tensor(binary_hbr_weights, dtype=torch.float32)
        #Reshape to N, C, H, W
        tensor_binary_hbr_weights = tensor_binary_hbr_weights.unsqueeze(0).unsqueeze(0)
        
        #Convolutional filter: 
        #Since the fNIRS data was sampled at 3.9 Hz, we averaged every 4 consecutive samples to approximate the data at a 1-second resolution
        # using a convolutional filter. 
        
        cnn = Conv2d(1,1, kernel_size=(1,4), stride=(1,4), padding=(0,1), padding_mode='replicate')
        cnn.weight.data.fill_(1.0/4.0)
        cnn.bias.data.fill_(0.0)
        
        
        convolved_hbo_weights = cnn(tensor_binary_hbo_weights)
        convolutional_hbr_weights = cnn(tensor_binary_hbr_weights)
        
        print("convolved hbo weights: ", convolved_hbo_weights.shape)
        print("convolved hbr weights: ", convolutional_hbr_weights.shape)
    
        convolved_hbo_weights = convolved_hbo_weights.squeeze(0).squeeze(0).detach().numpy()
    
        convolved_hbr_weights = convolutional_hbr_weights.squeeze(0).squeeze(0).detach().numpy()
        
        
        # We summed the SHAP values across all channels to obtain the relevance of each second
        summed_hbo_weights = np.sum(convolved_hbo_weights, axis=0)
        summed_hbr_weights = np.sum(convolved_hbr_weights, axis=0)

   
        # Save the relevant temporal points for each subject in another array
        count_times[count] = summed_hbo_weights
        count_times_hbr[count] = summed_hbr_weights

        
        # We summed the SHAP values across all  time points to obtain the relevance of each channel throughout the duration of the event
        summed_hbo_weights_ch = np.sum(convolved_hbo_weights, axis=1)
        summed_hbr_weights_ch = np.sum(convolved_hbr_weights, axis=1)
        

        
        count_channels[count] = summed_hbo_weights_ch
        count_channels_hbr[count] = summed_hbr_weights_ch
        
        count += 1
        
        
     
    # sum count times across all folders from the cross-validation    
    count_times = np.sum(count_times, axis=0)
    count_times_hbr = np.sum(count_times_hbr, axis=0)

    all_subjects_count_times[count_subjects] = count_times
    all_subjects_count_times_hbr[count_subjects] = count_times_hbr
    
    count_subjects += 1
    print(count_times)
    

    
    count_channels = np.sum(count_channels, axis=0)
    count_channels_hbr = np.sum(count_channels_hbr, axis=0)
    

    
    # Create a list of tuples with channel names, brain areas, and counts
    data = []
    data_hbr = []

    for brain_area, channels in channels_to_brain_areas.items():
        for channel in channels:
            if channel in hbo_names_o:
                index = hbo_names_o.index(channel)
                count_= count_channels[index]
                count_hbr = count_channels_hbr[index]
                data.append((channel, brain_area, count_))
                data_hbr.append((channel, brain_area, count_hbr))
                
            else:
                data.append((channel, brain_area, 0))
                data_hbr.append((channel, brain_area, 0))



    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=['Channel Name', 'Brain Area', 'Count'])
    
    # Combine brain areas that are the same and sum up the counts
    
    # The channels are aggregated by brain regions; therefore, the analysis will be conducted at the region level rather than the individual channel level
    df_combined = df.groupby('Brain Area').agg({'Count': 'sum'}).reset_index()

    df_hbr = pd.DataFrame(data_hbr, columns=['Channel Name', 'Brain Area', 'Count'])
    df_combined_hbr = df_hbr.groupby('Brain Area').agg({'Count': 'sum'}).reset_index()
    
    
    list_dataframes.append(df_combined)
    list_dataframes_hbr.append(df_combined_hbr)

    


# Create a grid of 2x4 subplots for count time histograms of all subjects
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each subject's count time histogram in a separate subplot
for i, (count_times, count_times_hbr) in enumerate(zip(all_subjects_count_times, all_subjects_count_times_hbr)):
    ax = axes[i]
    bar_width = 0.35
    index = np.arange(count_times.shape[0])
    
    ax.bar(index, count_times, bar_width, label='HbO', color='b')
    ax.bar(index + bar_width, count_times_hbr, bar_width, label='HbR', color='g')
    ax.set_title(f'sub.{i}', fontsize=23)
    # Show only sparse x-ticks, e.g., 1, 5, 10, 15
    sparse_ticks = [1, 5, 10, 15,17]
    ax.set_xticks(sparse_ticks)
    ax.set_xticklabels(sparse_ticks, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)  # Modify size of y-ticks labels
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set y-axis ticks as integers

# Set common labels
fig.text(0.5, 0.02, 'Time (s)', ha='center', va='center', fontsize=25)
fig.text(0.03, 0.5, 'SHAP values', ha='center', va='center', rotation='vertical', fontsize=24)

# Add a single legend for the entire figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=23, bbox_to_anchor=(0.52, 1.03))

# Adjust layout
plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

# Save the figure
plt.savefig(os.path.join(folder_, 'count_times_grid_shap.png'), dpi=500)

plt.show()



# compute average and standard deviation of all subjects count times
all_subjects_count_times_mean = np.mean(all_subjects_count_times, axis=0)
all_subjects_count_times_std = np.std(all_subjects_count_times, axis=0)

all_subjects_count_times_mean_hbr = np.mean(all_subjects_count_times_hbr, axis=0)
all_subjects_count_times_std_hbr = np.std(all_subjects_count_times_hbr, axis=0)


print("All subjects count times mean: ", all_subjects_count_times_mean)
print("All subjects count times std: ", all_subjects_count_times_std)




# Set a common figure size for both plots
figsize = (12, 8)

plt.figure(figsize=figsize)
bar_width = 0.35
index = np.arange(all_subjects_count_times_mean.shape[0])

plt.bar(index, all_subjects_count_times_mean, bar_width, yerr=all_subjects_count_times_std, label='HbO', color='b', capsize=5)
plt.bar(index + bar_width, all_subjects_count_times_mean_hbr, bar_width, yerr=all_subjects_count_times_std_hbr, label='HbR', color='g', capsize=5)

plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('SHAP values', fontsize=25)
plt.xticks(ticks=np.arange(0, 18, 1), fontsize=17)
plt.yticks(fontsize=17)
plt.legend(fontsize=23)
plt.tight_layout()
plt.savefig(os.path.join(folder_, 'all_subjects_count_times_mean_std_shap.png'), dpi=500)
plt.show()



#Concatenate all dataframes 

all_subjects_df = pd.concat(list_dataframes, ignore_index=True)
all_subjects_df_hbr = pd.concat(list_dataframes_hbr, ignore_index=True)



# Group by 'Brain Area' and calculate mean and std for 'Count'
all_subjects_df_mean = all_subjects_df.groupby('Brain Area').agg({'Count': 'mean'}).reset_index()
all_subjects_df_std = all_subjects_df.groupby('Brain Area').agg({'Count': 'std'}).reset_index()

all_subjects_df_hbr_mean = all_subjects_df_hbr.groupby('Brain Area').agg({'Count': 'mean'}).reset_index()
all_subjects_df_hbr_std = all_subjects_df_hbr.groupby('Brain Area').agg({'Count': 'std'}).reset_index()






# Plot double bar chart using all_subjects_df and all_subjects_df_hbr
plt.figure(figsize=(12, 8))  # Increase figure size for better readability
bar_width = 0.35
index = np.arange(len(all_subjects_df_mean['Brain Area']))

plt.bar(
    index,
    all_subjects_df_mean['Count'],
    bar_width,
    yerr=all_subjects_df_std['Count'],
    label='HbO',
    color='tab:brown',
    capsize=5,
    error_kw=dict(lw=3)  # Make error bars bolder
)
plt.bar(
    index + bar_width,
    all_subjects_df_hbr_mean['Count'],
    bar_width,
    yerr=all_subjects_df_hbr_std['Count'],
    label='HbR',
    color='orange',
    capsize=5,
    error_kw=dict(lw=3)  # Make error bars bolder
)

plt.xlabel('Brain Area', fontsize=25)
plt.ylabel('SHAP values', fontsize=24)
plt.xticks(ticks=index + bar_width / 2, labels=all_subjects_df_mean['Brain Area'], rotation=60, fontsize=25)
plt.yticks(fontsize=17)
plt.legend(fontsize=23)
plt.tight_layout()  # Adjust layout to fit everything
plt.savefig(os.path.join(folder_, 'all_subjects_brain_area_counts_mean_std_shap.png'), dpi=500)
plt.show()



# Save the combined DataFrame for all subjects to a CSV file
#all_subjects_df.to_csv(os.path.join(folder_, 'all_subjects_combined_brain_area_counts.csv'), index=False)

# Create a grid of 4x2 subplots for brain area counts of all subjects
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each subject's brain area counts in a separate subplot
for i, (df, df_hbr) in enumerate(zip(list_dataframes, list_dataframes_hbr)):
    ax = axes[i]
    bar_width = 0.35
    index = np.arange(len(df['Brain Area']))
    
    ax.bar(index, df['Count'], bar_width, label='HbO', color='tab:brown')
    ax.bar(index + bar_width, df_hbr['Count'], bar_width, label='HbR', color='orange')
    ax.set_title(f'sub.{i}',fontsize=23)
    ax.tick_params(axis='x', rotation=60, labelsize=18)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df['Brain Area'])
    ax.tick_params(axis='y', labelsize=18)  # Modify size of y-ticks labels

# Set common labels
fig.text(0.5, 0.02, 'Brain Area', ha='center', va='center',fontsize=23)
fig.text(0.03, 0.5, 'SHAP values', ha='center', va='center', rotation='vertical',fontsize=23)

# Add a single legend for the entire figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=23, bbox_to_anchor=(0.52, 1.03))

# Adjust layout
plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

# Save the figure
plt.savefig(os.path.join(folder_, 'brain_area_counts_grid_shap.png'), dpi=500)
plt.show()







# %%
