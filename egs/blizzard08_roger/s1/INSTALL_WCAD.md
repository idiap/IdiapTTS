## Install [WCAD](https://github.com/b-schnell/wcad)

Do the following steps to setup WCAD (not by default installed):
- Open *IdiapTTS/tools/compile_other_speech_tools.sh* 
- Set ``install_wcad=true`` and all others to ``false``
- Run ``bash tools/compile_other_speech_tools.sh`` from the root directory
- Switch to *tools/wcad/* and set the path in *path.cfg* to the *compute-kaldi-pitch-feats* binary (comes with [KALDI](https://kaldi-asr.org/)).