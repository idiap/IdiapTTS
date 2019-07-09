# INSTALL

To install IdiapTTS, `cd IdiapTTS` and run the below steps:

- Install basic tools, mainly for data preparation (the festival installation requires gcc-4.8, only needed for tts_frontend)  
  ``bash tools/compile_other_speech_tools.sh``  
  Also install HTK (requires account at http://htk.eng.cam.ac.uk/register.shtml) with  
  ``bash tools/compile_htk <HTK_USERNAME> <HTK_PASSWORD>``
- To make the TTS frontend working you have to download the unilex dictionary from [http://www.cstr.ed.ac.uk/projects/unisyn/](http://www.cstr.ed.ac.uk/projects/unisyn/) and copy the files from *festival/lib/dicts/unilex/* into your festival directory at the same location *full_festival_location/lib/dicts/unilex/*.

 For all the following we recommend to use a [Conda](https://docs.conda.io/en/latest/miniconda.html) environment with **Python 3.6** (bandmat does not build with pip in Python 3.7, it can be [built from source](https://github.com/MattShannon/bandmat/issues/11), May '19).
 
- Activate conda environment  
  - Install [PyTorch 0.4.1](https://pytorch.org/) with the appropriate cuda version. Example for CUDA8.0: ``conda install pytorch=0.4.1 cuda80 -c pytorch``
  - Install [Librosa](https://librosa.github.io/librosa/index.html)  
  ``conda install -c conda-forge librosa``
  - If you use conda make sure you got ``pip`` installed in your environment, then install IdiapTTS by running  
  ``pip install .``
