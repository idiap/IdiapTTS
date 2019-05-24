#!/usr/bin/env bash
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

convertsecs() {
 h=$(bc <<< "${1}/3600")
 m=$(bc <<< "(${1}%3600)/60")
 s=$(bc <<< "${1}%60")
 printf "%02d:%02d:%05.2f\n" $h $m $s
}

dir_audio=${1:-"."}

if [ "$#" -eq 2 ]; then
    file_id_list=${2:-"file_id_list.txt"}
    echo "Load files from ${file_id_list} and search them in ${dir_audio}."
    IFS=$'\r\n' GLOBIGNORE='*' command eval 'utts=($(cat ${file_id_list}))'
else
    echo "Find files in ${dir_audio}."
    cwd=$PWD
    cd "${dir_audio}"
    # find . -name "*.wav" | xargs echo
    utts=()
    while IFS=  read -r -d $'\0'; do
        utts+=("$REPLY")
    done < <(find . -name "*.wav" -print0 )
    cd "${cwd}"
fi
#echo ${utts[@]:0:10}

total_length=0
for filename in "${utts[@]}"; do
    length=$(soxi -D "${dir_audio}/${filename%.*}.wav")
    total_length=$(echo "${total_length} + ${length}" | bc)
done

echo $(convertsecs ${total_length})