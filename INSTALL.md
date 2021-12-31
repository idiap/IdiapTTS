# INSTALL

To install IdiapTTS, `cd IdiapTTS` and run the below steps:

- Install basic tools, mainly for data preparation (the festival installation requires gcc-4.8, only needed for tts_frontend)
  ``bash tools/compile_other_speech_tools.sh``
  Also install HTK (requires account at http://htk.eng.cam.ac.uk/register.shtml) with
  ``bash tools/compile_htk <HTK_USERNAME> <HTK_PASSWORD>``
- To make the TTS frontend working you have to download the unilex dictionary from [http://www.cstr.ed.ac.uk/projects/unisyn/](http://www.cstr.ed.ac.uk/projects/unisyn/) and copy the files from *festival/lib/dicts/unilex/* into your festival directory at the same location *full_festival_location/lib/dicts/unilex/*.
- Ensure that bc and soxi packages are installed in your shell.

 For all the following we recommend to use a [Conda](https://docs.conda.io/en/latest/miniconda.html) environment with **Python 3.6** (bandmat does not build with pip in Python 3.7, it can be [built from source](https://github.com/MattShannon/bandmat/issues/11), May '19).

- Activate conda environment
  - Install [PyTorch 1.6.0](https://pytorch.org/) with the appropriate cuda version. Example for CUDA8.0: ``conda install pytorch=1.6.0 cuda80 -c pytorch``
  - If you use conda make sure you got ``pip`` installed in your environment, then install IdiapTTS by running
  ``pip install .`` or use ``pip install -e .`` to install in editable mode.
