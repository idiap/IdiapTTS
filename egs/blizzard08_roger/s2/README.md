**Requirements:**
- Install the main requirements by follow the installation process in the *INSTALL.md* file in the root directory.  
- Install WCAD by following the installation process in *../s1/INSTALL_WCAD.md*?  
- Install the *neural_filters* package from this directory with:  
  ``pip install -r requirements.txt``  
- Run ``source cmd.sh``  


# Generate features

### 1. Create the database
Run `./01_setup.sh --max_length_sec 15 <Path to roger db> <num_workers>` where the path to the roger database should point to the folder containing the *utts.data* file and *wav* folder. The script creates links to the database of those audio files which are less than 15 seconds long. It then removes all silence but 200 ms in the front and back of each file. This behaviour can be stopped by giving the `--no_silence_removal` flag. Or the length can be changed by giving `--min_silence_ms <integer>` to the call of `silence_remove.py`. The script creates a *file_id_list_all.txt* with all ids, a *file_id_list_full.txt* with the ids from carroll, arctic, and theherald1-3, and a *file_id_list_demo.txt* with the ids from theherald1.

The following errors can be ignored:
* `cp: cannot stat '/IdiapTTS/egs/blizzard08_roger/s2/database/wav_org_silence/*': No such file or directory`
* ``./01_setup.sh: line 184: syntax error near unexpected token `|'``  
  ``./01_setup.sh: line 184: ` | sort) > "${file_id_list_all}"'``

***
NOTE: The following sequences of steps can be run in parallel (2, 3), (4, 5), and 6.

### 2. Create force-aligned HTK full labels
Run `./02_prepare_HTK_labels_en_am.sh full <num_workers>` to create force-aligned HTK full labels for the **full** id list. The script uses the Idiap TTS frontend (which relies on a festival backend) to generate phoneme labels with 5 states per phoneme. It assumes an English speaker with an American accent. It then uses HTK to force-align the labels to the audio.

### 3. Questions
Run `./03_prepare_question_labels.sh full <num_workers>` to generate questions from the HTK full labels for the **full** id list. The labels are at 5 ms frames and identically for consecutive frames of the same phoneme state.

### 4. Acoustic features
Run `./04_prepare_WORLD_labels.sh full <num_workers>` to extract acoustic features (60 MGC, LF0, VUV, 1 BAP) with [WORLD](https://github.com/mmorise/World) / [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) from the audio files specified in the **full** id list. Each feature is saved in a subdirectory. The number of spectral features can be specified in the script call as well. The LF0 features are later used to train the models.

### 5. Acoustic features with deltas and double deltas
Run `./05_prepare_WORLD_deltas_labels.sh full <num_workers>` to extract acoustic features as in **4.** but also compute deltas and double deltas for MGC, LF0, and BAP and save them all together in a *.cmp* file. These features are later used to train the baseline model.

### 6. Extract WCAD atoms
Run `./06_prepare_atom_labels.sh full <num_workers> database/file_id_list_full.txt` to use the [WCAD](https://github.com/b-schnell/wcad) atom extractor to extract phrase and regular atoms for the **full** id list. The atoms are stored in *.atoms* files while the phrases are in the *.phrase* files. The current implementation of the WCAD atom extractor can only extract a single phrase atom, therefore it fails for very long utterances. However, the extractor creates a *wcad_file_id_list_\<voice>.txt* in the database folder which only contains the ids for which the extraction seemed to have worked. Feature files for all the samples will still be created. To remove the excluded features one can delete the feature folder and run the script with the *wcad_file_id_list_\<voice>.txt* again. A summary of warnings is saved in the feature director at *log/file_id_list_\<voice>_WARNINGS.txt*, where one can check the ids of the failed samples. For training only the *wcad_file_id_list_<voice>.txt* should be used.


# Training

### Baseline model
Train the baseline model by running `python BaselineTrainer.py`. The class contains all the hyper-parameters used for our experiments. It uses a bi-LSTM based acoustic model to predict acoustic features with deltas and double deltas.

### PhraseAtomNeuralFilters model
To train the proposed model run `python MyPhraseAtomNeuralFiltersTrainer.py`. The class contains all the hyper-parameters used for our experiments. It will first train the pre-net with the AtomLoss to predict atom spikes.
 Then a NeuralFilters layer is stacked on top of the trained pre-net and everything is trained end-to-end with a weighted MSE on the target LF0. The bias is initialized by the mean LF0 value computed from the data, which is then also trained.