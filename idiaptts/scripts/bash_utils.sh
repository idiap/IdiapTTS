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

    echo "Copy unchanged files in downsample step..."
    cp -R -u -p "${dir_audio}_org/"* "${dir_audio}"  # -u copy only when source is newer than destination file or if is missing, -p preserve mode, ownership, timestamps etc.
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

    # python ${dir_src}/data_preparation/audio/silence_remove.py --dir_wav "${dir_audio}_org_silence"/ --dir_out "${dir_audio}/" --file_id_list "database/file_id_list_full.txt" --silence_db ${silence_db} --min_silence_ms ${min_silence_ms}

    ./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_silence_removal_blockJOB.log \
         ${dir_src}/data_preparation/audio/silence_remove.py \
                --dir_wav ${dir_audio}_org_silence/ \
                --dir_out ${dir_audio}/ \
                --file_id_list ${dir_data}/${name_file_id_list}_blockJOB \
                --silence_db ${silence_db} \
                --min_silence_ms ${min_silence_ms}

    # Copy files not touched in this remove silence step.
    echo "Copy unchanged files in remove silence step..."
    cp -R -u -p "${dir_audio}_org_silence/"* "${dir_audio}/"  # -u copy only when source is newer than destination file or if it is missing, -p preserve mode, ownership, timestamps etc.

    # Remove intermediate files.
    rm -r -f "${dir_audio}_org_silence"/

    # Print errors and save warnings.
    eval grep --ignore-case "WARNING" "${dir_logs}/${name_file_id_list}_silence_removal_block{1..${num_blocks}}.log" >| "${dir_logs}/${name_file_id_list}_silence_removal_WARNINGS.txt"
    eval grep --ignore-case "ERROR" "${dir_logs}/${name_file_id_list}_silence_removal_block{1..${num_blocks}}.log"
}

normalize_loudness() {
    ref_rms=${1:-"0.1"}

    echo "Normalize loudness..."

    rm -r -f "${dir_audio}_org_loudness"/
    mv "${dir_audio}" "${dir_audio}_org_loudness"/
    mkdir -p "${dir_audio}"

    ./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/loudness_normalization_${name_file_id_list}_blockJOB.log \
         ${dir_src}/data_preparation/audio/normalize_loudness.py \
                --dir_wav ${dir_audio}_org_loudness/ \
                --dir_out ${dir_audio}/ \
                --file_id_list ${dir_data}/${name_file_id_list}_blockJOB \
                --ref_rms ${ref_rms}

    echo "Copy unchanged files in normalize loudness step..."
    cp -R -u -p "${dir_audio}_org_loudness/"* "${dir_audio}/"  # -u copy only when source is newer than destination file or if it is missing, -p preserve mode, ownership, timestamps etc.

    # Remove intermediate files.
    rm -r -f "${dir_audio}_org_loudness"/

    # Print errors and save warnings.
    eval grep --ignore-case "WARNING" "${dir_logs}/loudness_normalization_${name_file_id_list}_block{1..${num_blocks}}.log" >| "${dir_logs}/loudness_normalization_${name_file_id_list}_WARNINGS.txt"
    eval grep --ignore-case "ERROR" "${dir_logs}/loudness_normalization_${name_file_id_list}_block{1..${num_blocks}}.log"
}

high_pass_filter() {

    stop_freq_Hz=${1:-"70"}
    pass_freq_Hz=${2:-"100"}
    filter_order=${3:-"1001"}

    echo "High pass filtering..."

    rm -r -f "${dir_audio}_org_signal"/
    mv "${dir_audio}" "${dir_audio}_org_signal"/
    mkdir -p "${dir_audio}"

    ./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/high_pass_filtering_${name_file_id_list}_blockJOB.log \
         ${dir_src}/data_preparation/audio/high_pass_filter.py \
                --dir_wav ${dir_audio}_org_signal/ \
                --dir_out ${dir_audio}/ \
                --file_id_list ${dir_data}/${name_file_id_list}_blockJOB \
                --stop_freq_Hz ${stop_freq_Hz} \
                --pass_freq_Hz ${pass_freq_Hz} \
                --filter_order ${filter_order}

    echo "Copy unchanged files in high pass filtering step..."
    cp -R -u -p "${dir_audio}_org_signal/"* "${dir_audio}/"  # -u copy only when source is newer than destination file or if it is missing, -p preserve mode, ownership, timestamps etc.

    # Remove intermediate files.
    rm -r -f "${dir_audio}_org_signal"/

    # Print errors and save warnings.
    eval grep --ignore-case "WARNING" "${dir_logs}/high_pass_filtering_${name_file_id_list}_block{1..${num_blocks}}.log" >| "${dir_logs}/high_pass_filtering_${name_file_id_list}_WARNINGS.txt"
    eval grep --ignore-case "ERROR" "${dir_logs}/high_pass_filtering_${name_file_id_list}_block{1..${num_blocks}}.log"
}

single_channel_noise_reduction() {

    echo "Single channel noise reduction..."

    rm -r -f "${dir_audio}_org_noisy"/
    mv "${dir_audio}" "${dir_audio}_org_noisy"/
    mkdir -p "${dir_audio}"

    python ${dir_src}/data_preparation/audio/single_channel_noise_reduction.py \
                  --dir_wav ${dir_audio}_org_noisy/ \
                  --dir_out ${dir_audio}/ \
                  --file_id_list ${dir_data}/${name_file_id_list}

# # Matlab cannot be run easily on the grid.
#    ./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/single_channel_noise_reduction_${name_file_id_list}_blockJOB.log \
#         ${dir_src}/data_preparation/audio/single_channel_noise_reduction.py \
#                --dir_wav ${dir_audio}_org_noisy/ \
#                --dir_out ${dir_audio}/ \
#                --file_id_list ${dir_data}/${name_file_id_list}_blockJOB

    echo "Copy unchanged files in single channel noise reduction step..."
    cp -R -u -p "${dir_audio}_org_noisy/"* "${dir_audio}/"  # -u copy only when source is newer than destination file or if it is missing, -p preserve mode, ownership, timestamps etc.

    # Remove intermediate files.
#    rm -r -f "${dir_audio}_org_noisy"/

    # Print errors and save warnings.
    eval grep --ignore-case "WARNING" "${dir_logs}/single_channel_noise_reduction_${name_file_id_list}_block{1..${num_blocks}}.log" >| "${dir_logs}/single_channel_noise_reduction_${name_file_id_list}_WARNINGS.txt"
    eval grep --ignore-case "ERROR" "${dir_logs}/single_channel_noise_reduction_${name_file_id_list}_block{1..${num_blocks}}.log"
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