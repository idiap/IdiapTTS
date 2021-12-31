#!/usr/bin/env bash

remove_dur(){
    local file_id_list="${1}"
    local dir_labels="${2}"

    mkdir -p "${dir_labels}_no_align/"

    IFS=$'\r\n' GLOBIGNORE='*' command eval 'utts=($(cat ${file_id_list}))'

    for file_id in ${utts[@]}; do
        utt_id=$(basename "${file_id}")  # Remove possible speaker folder in path.
        subfolder_name=$(basename "${dir_labels}")
        cp ${dir_labels}/${utt_id}.lab ${dir_labels}_no_align/
        sed -i 's/[ ]*[^ ]*[ ]*[^ ]* //' ${dir_labels}_no_align/${utt_id}.lab
    done
}

file_id_list="${1}"
dir_labels="${2}"

echo "Remove durations..."

remove_dur "${file_id_list}" "${dir_labels}/full"
remove_dur "${file_id_list}" "${dir_labels}/mono"
