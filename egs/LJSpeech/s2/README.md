**Requirements:**
- Install the main requirements by follow the installation process in the *INSTALL.md* file in the root directory.  
- Run ``source cmd.sh``  
- Install r9y9's WaveNet implementation (only the library) by running  
`pip install wavenet_vocoder` 


# Generate features

### 1. Create the database
Run `./01_setup.sh <Path to LJSpeech db> <num_workers>` where the path to the LJSpeech database should point to the folder containing the *metadata.csv* file and *wavs* folder. The script creates links to the audio files in the database and stores them in *wav/*. It then removes all silence but 200 ms in the front and back of each file. This behaviour can be stopped by giving the `--no_silence_removal` flag. Or the length can be changed by giving `--min_silence_ms <integer>` to the call of `silence_remove.py`. It also down-samples all files to the `target_frame_rate`, which is 16 kHz by default (note that changing this requires also changing it in the hyper-parameters of all model trainers). The script creates a *file_id_list_full.txt* with all ids, and a *file_id_list_demo.txt* which contains 300 (seeded) randomly picked ids out of all.

The following errors can be ignored:
* `cp: cannot stat '/IdiapTTS/egs/LJSpeech/s1/database/wav_org/*': No such file or directory`
* `cp: cannot stat '/IdiapTTS/egs/blizzard08_roger/s1/database/wav_org_silence/*': No such file or directory`
* ``./01_setup.sh: line 184: syntax error near unexpected token `|'``  
  ``./01_setup.sh: line 184: ` | sort) > "${file_id_list_all}"'``

***
NOTE: The following steps 2, 3, and 4 need to be run sequentially, while step 5 can be run in parallel to them.

### 2. Create phoneme labels
Run `./02_prepare_phoneme_labels_en_am.sh full <num_workers>` to create not-aligned HTK full labels for the **full** id list. The script uses the Idiap TTS frontend (which relies on a festival backend) to generate phoneme labels with 5 states per phoneme. It assumes an English speaker with an American accent. Training encoder-attention-decoder models from phonemes instead of characters significantly improves the learning speed at very low cost.

### 3. Acoustic features
Run `./05_prepare_WORLD_labels.sh full <num_workers> ` to extract acoustic features (60 MGC, LF0, VUV, 1 BAP) with [WORLD](https://github.com/mmorise/World) / [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) from the audio files specified in the **full** id list. The features are saved separately in different subdirectories in the *WORLD/* directory. 



# Training

### E2E phoneme to acoustic features model
TODO

### WaveNet
This framework uses the WaveNet [implementation](https://github.com/r9y9/wavenet_vocoder) of Ryuichi (r9y9) Yamamoto via a wrapper class. It is trained by running `python MyWaveNetVocoderTrainer.py`. The class contains similar hyper-parameters than those reported by Yamamoto. The default configuration uses 256 bit mu-law quantisation with 12 layers divided into 2 stacks with a kernel size of 2. The MoL configuration is commented out, but can be used instead of mu-law quantisation. However, the MoL configuration has lead to background noise until now.

# Synthesis
TODO