**Requirements:**
- Install the main requirements by follow the installation process in the *INSTALL.md* file in the root directory.  
- Run ``source cmd.sh``  


# Generate features

### 1. Create the database
Run `./01_setup.sh <Path to VCTK db> <num_workers>` where the path to the VCTK database should point to the folder containing the *speaker-info.txt* file and *wav48* folder. The script creates links to the audio files in the database and stores them in *wav/*. It then removes all silence but 10 ms in the front and back of each file. This behaviour can be stopped by giving the `--no_silence_removal` flag. Or the length can be changed by giving `--min_silence_ms <integer>` to the call of `silence_remove.py`. It also down-samples all files to the `target_frame_rate`, which is 16 kHz by default (note that changing this requires also changing it in the hyper-parameters of all model trainers). The script creates a *file_id_list_full.txt* with all ids, a *file_id_list_demo.txt* with speakers p225, p226, p227, and p269, a *file_id_list_half.txt* with the first 55 speakers in the *speaker-info.txt* file, and a *file_id_list_English.txt* which contains only the speakers which have an English accent specified in the *speaker-info.txt*.

The following errors can be ignored:
* `cp: cannot stat '/IdiapTTS/egs/VCTK/s1/database/wav_org/*': No such file or directory`
* `cp: cannot stat '/IdiapTTS/egs/VCTK/s1/database/wav_org_silence/*': No such file or directory`

***
NOTE: The following steps 2, 3, and 4 need to be run sequentially, while step 5 can be run in parallel to them.

### 2. Create force-aligned HTK full labels
Run `./02_prepare_HTK_labels_en_br.sh English <num_workers>` to create force-aligned HTK full labels for the **English** id list. The script uses the Idiap TTS frontend (which relies on a festival backend) to generate phoneme labels with 5 states per phoneme. It assumes an English speaker with a british accent. It then uses HTK to force-align the labels to the audio.

### 3. Questions
Run `./03_prepare_question_labels.sh --multispeaker full <num_workers>` to generate questions from the HTK full labels for the **full** id list. The labels are at 5 ms frames and identically for consecutive frames of the same phoneme state. The `--multispeaker` flag tells the script to expect audio from multiple speakers so that it creates speaker-dependent alignments.

### 4. Splitting speakers for adaptation
Run `python 04_create_adapt_list.py` to create...

### 5. Acoustic features with deltas and double deltas
Run `./05_prepare_WORLD_deltas_labels.sh full <num_workers> database/file_id_list_English.txt 30` to extract acoustic features (30 MGC, LF0, VUV, 1 BAP) with their deltas and double deltas (except for VUV) with [WORLD](https://github.com/mmorise/World) / [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) from the audio files specified in the **English** id list. All features are saved together in the *WORLD/cmp_mgc30* directory as *.cmp* files. 



# Training

### Baseline model

### VTLN model


# Adaptation