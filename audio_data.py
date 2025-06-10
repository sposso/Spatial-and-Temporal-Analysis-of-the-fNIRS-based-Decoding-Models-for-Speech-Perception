import mne_nirs
import mne
from mne_bids import BIDSPath, get_entity_vals
from mne.preprocessing.nirs import optical_density, temporal_derivative_distribution_repair
from itertools import compress
import numpy as np
import matplotlib.pyplot as plt


def epoch_preprocessing(subject_index, t_min,t_max):
    
    """
    Arguments:
    Subject index: int = Index of the subject to be processed [0, 1, 2, 3, 4, 5, 6, 7]
    t_min: float =  Time to initialize the epoch
    t_max: float = Time to end the epoch
    
    Returns:
    haemo:= The preprocessed haemoglobin data
    epochs: mne.Epochs object = The preprocessed epochs. You can use this to get the data and labels for the classification task.
    X: np.array = The preprocessed data. Shape: (n_trials, n_channels, n_times)
    Y: np.array = The labels for the classification task. Shape: (n_trials,)
    Y labels : 1.0-> Audio, 3.0 -> Control (Silence)
    """
    
    root = mne_nirs.datasets.audio_or_visual_speech.data_path()
    
    print(root)
    
    subject= get_entity_vals(root, "subject")[subject_index]
   
        
    dataset = BIDSPath(
            root=root,
            suffix="nirs",
            extension=".snirf",
            subject=subject,
            task="AudioVisualBroadVsRestricted",
            datatype="nirs",
            session="01"
                            )
                
    #Raw intensity data
    raw_intensity = mne.io.read_raw_snirf(dataset.fpath)
    
    raw_intensity.annotations.rename(
        {"1.0": "Audio", "2.0": "Video", "3.0": "Control", "15.0": "Ends"}
    )
    
    
    # Converting raw intensity to optical density
    raw_od = optical_density(raw_intensity)


    #Compute scalp coupling index to identify optodes that were not well attached to the scalp
    #Rejection criterion of < 0.8
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.8))
    
    print("Number of channels before removing bad channels: ", len(raw_od.ch_names))
    #raw_od = raw_od.info.pop('bads')

    raw_od = raw_od.drop_channels(raw_od.info['bads'])

    print("Number of channels after removing bad channels: ", len(raw_od.ch_names))
        

    
    #Temporal Derivative Distribution Repair
    
    corrected_tddr = temporal_derivative_distribution_repair(raw_od)
    
    #Apply short channel correction  to remove the influence from non-cortical changes in blood oxygenation.
    
    # A short separation channels measures solely the extracerebral signals, which includes 
    # blood presure waves, mayer waves, respiration and cardiac cycles.
    # The signal components od the short separation channel can be seen as the noise in the signal of the 
    # long channel.BY removing these components from the log channel, you cna minimize the noise.
    
    
    od_corrected = mne_nirs.signal_enhancement.short_channel_regression(corrected_tddr)
    #Convert optical density to haemoglobin concentration using the Beer-Lambert Law
    haemo = mne.preprocessing.nirs.beer_lambert_law(od_corrected, ppf=6)
    
    haemo = mne_nirs.channels.get_long_channels(haemo)
    
    #Bandpass filter the haemoglobin data between 0.02 and 0.4 Hz
    #to removoe slow drifts and components related to the heart rate
    haemo = haemo.filter(0.02,0.4)

    #plot haemo

    
    #haemo.plot(n_channels=1, duration= 32, show_scrollbars=False)
    
    
    #Signal enhancement method (negative correlation enhancement algorithm) Cui et. al. 2010
    
    haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(haemo)
    

        
    events, event_dict = mne.events_from_annotations(haemo)
    #Epochs corresponding to 8 s before the stimulus onset and 30 s after the stimulus onset
    #An epoch rejection criterion  was employed to exclude epochs with a signal amplitude > 100 uM
    epochs = mne.Epochs(
        haemo,
        events,
        event_id=event_dict,
        tmin=t_min,
        tmax=t_max,
        reject=dict(hbo=100e-6, hbr=100e-6), # Epoc rejection criterion
        reject_by_annotation=True,
        proj=False,
        baseline=None,
        detrend=None,
        preload=True,
        verbose=True,
    )
    
    # I am just selecting the Audio and Control epochs
    epochs = epochs[["Audio", "Control"]]
    
    return haemo, epochs


def final_audio_data(subject_index,t_min,t_max,chroma=None):
    
    """ Function to pick the chroma data from the haemoglobin data:
    chroma: str -> ['hbo', 'hbr']
        The chroma data to be picked from the haemoglobin data
        
    subject_index: int
        
        
    Returns: feature matrix X and target vector Y
    
    X shape: (n_trials, n_channels, n_times)
    
    Y shape: (n_trials,)
    Y labels : -1 -> Audio, 1 -> Control (Silence)
    Epochs

    """
    
    if chroma is None:
        
        print("Hbo and hbr data will be returned together")
        
        _,epochs = epoch_preprocessing(subject_index, t_min, t_max)
        
        
        X= epochs.get_data()
        Y = epochs.events[:, 2]
        
        map = {1: -1, 2: 1}

        Y = [map[y] for y in Y]
        Y = np.array(Y)
       
        
    else:
        
        _, epochs = epoch_preprocessing(subject_index,t_min,t_max)
        epochs.pick(chroma)
        
        X = epochs.get_data()
        
        Y = epochs.events[:, 2]
        
        map = {1: -1, 2: 1}

        Y = [map[y] for y in Y]
        Y = np.array(Y)
        
    return X, Y,epochs
