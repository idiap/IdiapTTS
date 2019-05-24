#!/bin/bash

# Taken from Merlin (https://github.com/CSTR-Edinburgh/merlin)

#########################################
######### Install Dependencies ##########
#########################################
#sudo apt-get -y install libncurses5 libncurses5-dev libcurses-ocaml # for sudo users only

current_working_dir=$(pwd)
tools_dir=${current_working_dir}/$(dirname $0)
cd $tools_dir

install_speech_tools=true
install_festival=true
install_festvox=true
install_tts_frontend=true
install_wcad=false

# 1. Get and compile speech tools
if [ "$install_speech_tools" = true ]; then
    echo "downloading speech tools..."
    speech_tools_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-release.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $speech_tools_url
    elif hash wget 2>/dev/null; then
        wget $speech_tools_url
    else
        echo "please download speech tools from $speech_tools_url"
        exit 1
    fi
    tar xzf speech_tools-2.4-release.tar.gz

    echo "compiling speech tools..."
    (
        cd speech_tools;
        ./configure;
        make;
        make install
    )

fi

# export paths
export ESTDIR=$tools_dir/speech_tools
export LD_LIBRARY_PATH=$ESTDIR/lib:$LD_LIBRARY_PATH
export PATH=$ESTDIR/bin:$PATH

# 2. Get and compile festival, download dicts and some voices
if [ "$install_festival" = true ]; then  # TODO: Update to speech-tools 2.5 and festival 2.5 (https://github.com/festvox/festival)
    echo "downloading festival..."
    festival_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/festival-2.4-release.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $festival_url
    elif hash wget 2>/dev/null; then
        wget $festival_url
    else
        echo "please download Festival from $festival_url"
        exit 1
    fi
    tar xzf festival-2.4-release.tar.gz

    echo "compiling festival..."
    (
        cd festival;
        ./configure;
        make CXX=gcc-4.8;
        make install
    )

    echo "downloading some useful lexicons..."
    dict1_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_CMU.tar.gz
    dict2_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_OALD.tar.gz
    dict3_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_POSLEX.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $dict1_url
        curl -L -O $dict2_url
        curl -L -O $dict3_url
    elif hash wget 2>/dev/null; then
        wget $dict1_url
        wget $dict2_url
        wget $dict3_url
    else
        echo "please download dictionaries from $festival_url"
        exit 1
    fi
    tar xzf festlex_CMU.tar.gz
    tar xzf festlex_OALD.tar.gz
    tar xzf festlex_POSLEX.tar.gz

    echo "downloading some voices for English..."
    festival_voice_url=http://festvox.org/packed/festival/2.4/voices
    voice1_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_kallpc16k.tar.gz
    voice2_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_rablpc16k.tar.gz
    voice3_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_cmu_us_slt_cg.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $voice1_url
        curl -L -O $voice2_url
        curl -L -O $voice3_url
    elif hash wget 2>/dev/null; then
        wget $voice1_url
        wget $voice2_url
        wget $voice3_url
    else
        echo "please download Festival voices from $festival_voice_url"
        exit 1
    fi
    tar xzf festvox_kallpc16k.tar.gz
    tar xzf festvox_rablpc16k.tar.gz
    tar xzf festvox_cmu_us_slt_cg.tar.gz

    if [ "$install_tts_frontend" = true ]; then

        echo "install tts_frontend..."
        (
            cd tts_frontend;
            ./install ../festival/
        )        
    fi
fi

# export paths
export FESTDIR=$tools_dir/festival
export PATH=$FESTDIR/bin:$PATH

# 3. Get and compile festvox
if [ "$install_festvox" = true ]; then
    echo "downloading festvox..."
    festvox_url=http://festvox.org/festvox-2.7/festvox-2.7.0-release.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $festvox_url
    elif hash wget 2>/dev/null; then
        wget $festvox_url
    else
        echo "please download festvox from $festvox_url"
        exit 1
    fi
    tar xzf festvox-2.7.0-release.tar.gz

    echo "compiling festvox..."
    (
        cd festvox;
        ./configure;
        make;
    )

fi

# export paths
export FESTVOXDIR=$tools_dir/festvox

if [ "$install_wcad" = true ]; then
    git clone https://github.com/b-schnell/wcad.git
fi

echo "deleting downloaded tar files..."
rm -rf $tools_dir/*.tar.gz

if [ "$install_speech_tools" == true ] && [[ ! -f ${ESTDIR}/bin/ch_track ]]; then
    echo "Error installing speech tools"
    exit 1
elif [ "$install_festival" == true ] && [[ ! -f ${FESTDIR}/bin/festival ]]; then
    echo "Error installing Festival"
    exit 1
elif [ "$install_festvox" == true ] && [[ ! -f ${FESTVOXDIR}/src/vc/build_transform ]]; then
    echo "Error installing Festvox"
    exit 1
elif [ "$install_tts_frontend" == true ] && [ ! -d tts_frontend ]; then
    echo "Error installing tts_frontend."
    exit 1
elif [ "$install_wcad" == true ] && [ ! -d wcad ]; then
    echo "Error installing wcad."
    exit 1
else
    echo "All tools successfully compiled!!"
fi

