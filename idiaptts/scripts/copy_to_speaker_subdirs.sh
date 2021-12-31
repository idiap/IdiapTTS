#!/usr/bin/env bash

file_id_list="${1}"
dir_labels="${2}"

IFS=$'\r\n' GLOBIGNORE='*' command eval 'utts=($(cat ${file_id_list}))'

for file_id in ${utts[@]}; do
    utt_id=$(basename "${file_id}")  # Remove possible speaker folder in path.
    subfolder_name=$(basename "${dir_labels}")

    if [[ "${file_id}" == *\/* ]]; then  # File id contains a directory.
        speaker_id=${file_id%%/*}
    #        echo $file_id $speaker_id
        if [ -n ${speaker_id} ]; then  # If speaker id is not empty.
            utt_id=${file_id##*/}
    #        echo Copy ${utt_id} to ${speaker_id}/${utt_id}
            mkdir -p ${dir_labels}/${speaker_id}
            # Alignment script requires the files in speaker specific
            # subdirectories so copy them here. Don't move them
            # because model trainers require them to be in the main
            # directory. TODO: Can we remove this requirement?
            cp ${dir_labels}/${utt_id}.lab  ${dir_labels}/${speaker_id}/
        fi
    fi
done