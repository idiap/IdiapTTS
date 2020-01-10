#!/usr/bin/env bash

split_file_id_list() {
    file_id_list=${1:-${file_id_list}}
    name_file_id_list=${2:-$(basename ${file_id_list})}

     # Split into working blocks.
    if [ "$num_blocks" -gt "99" ]; then
        suffix_length=3
    elif [ "$num_blocks" -gt "9" ]; then
        suffix_length=2
    else
        suffix_length=1
    fi
    # Split file_id_list in the same way as utts_selected.data.
    split --numeric=1 --suffix-length ${suffix_length} -l ${block_size} ${file_id_list} ${dir_data}/${name_file_id_list}_block
    # Remove leading zeros in block numbering.
    for FILE in $(ls ${dir_data}/${name_file_id_list}_block*); do
        mv "${FILE}" "$(echo ${FILE} | sed -e 's:_block0*:_block:')" 2>/dev/null
    done
}

cleanup_split_file_id_list() {
    name_file_id_list=${1:-$(basename ${file_id_list})}
    # eval command required because otherwise brace expansion { .. } happens before $ expansion
    eval rm -f ${dir_data}/${name_file_id_list}_block{0..${num_blocks}}
}

downsample() {
    echo "Downsample files to frame rate: ${target_frame_rate} ..."
    rm -r -f "${dir_audio}_org"
    mv "${dir_audio}" "${dir_audio}_org"
    mkdir -p "${dir_audio}"

    ./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_down_sampling_blockJOB.log \
                    ${dir_src}/data_preparation/audio/down_sampling.py \
                    ${dir_audio}_org/ \
                    ${dir_audio}/ \
                    ${dir_data}/${name_file_id_list}_blockJOB \
                    ${target_frame_rate}

    cp -R -u -p "${dir_audio}_org/*" "${dir_audio}"  # -u copy only when source is newer than destination file or if is missing, -p preserve mode, ownership, timestamps etc.
    rm -r -f "${dir_audio}_org"/

    # Print errors and warnings.
    eval grep --ignore-case "WARNING" "${dir_logs}/${name_file_id_list}_down_sampling_block{1..${num_blocks}}.log"
    eval grep --ignore-case "ERROR" "${dir_logs}/${name_file_id_list}_down_sampling_block{1..${num_blocks}}.log"
}

remove_silence() {
    silence_db=${1:-"-30"}
    min_silence_ms=${2:-"200"}

    echo "Remove silence..."

    rm -r -f "${dir_audio}_org_silence"/
    mv "${dir_audio}" "${dir_audio}_org_silence"/
    mkdir -p "${dir_audio}"

    # python ${dir_src}/data_preparation/audio/silence_remove.py --dir_wav "${dir_audio}_org_silence"/ --dir_out "${dir_audio}/" --file_id_list "database/file_id_list_full.txt"

    ./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_silence_removal_blockJOB.log \
         ${dir_src}/data_preparation/audio/silence_remove.py \
                --dir_wav ${dir_audio}_org_silence/ \
                --dir_out ${dir_audio}/ \
                --file_id_list ${dir_data}/${name_file_id_list}_blockJOB \
                --silence_db ${silence_db} \
                --min_silence_ms ${min_silence_ms}

    # Copy files not touched in this remove silence step.
    cp -R -u -p "${dir_audio}_org_silence/*" "${dir_audio}/"  # -u copy only when source is newer than destination file or if is missing, -p preserve mode, ownership, timestamps etc.

    # Remove intermediate files.
    rm -r -f "${dir_audio}_org_silence"/

    # Print errors and save warnings.
    eval grep --ignore-case "WARNING" "${dir_logs}/${name_file_id_list}_silence_removal_block{1..${num_blocks}}.log" >| "${dir_logs}/${name_file_id_list}_silence_removal_WARNINGS.txt"
    eval grep --ignore-case "ERROR" "${dir_logs}/${name_file_id_list}_silence_removal_block{1..${num_blocks}}.log"
}

remove_long_files() {
    echo "Removing audio files longer than ${max_length_sec} seconds..."

    # Remove too long files.
    # TODO: Call it on the grid?
    num_removed=0
    for filename in "${utts_full[@]}"; do
        length=$(soxi -D "${dir_audio}/${filename}.wav")
        if (( $(echo "${length} > ${max_length_sec}" | bc -l) )); then
            rm "${dir_audio}/${filename}.wav"
            ((num_removed++))
        fi
    done
    echo "Removed ${num_removed} files."
}