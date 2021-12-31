#!/usr/bin/env bash

# A script to create the fixtures based on a file_id_list.txt from an egs folder.

###############################
# Default options and functions
#
# set -o xtrace # Prints every command before running it, same as "set -x".
set -o errexit # Exit when a command fails, same as "set -e".
            # Use "|| true" for those who are allowed to fail.
            # Disable (set +e) this mode if you want to know a nonzero return value.
set -o pipefail # Catch mysqldump fails.
set -o nounset # Exit when using undeclared variables, same as "set -u".
set -o noclobber # Prevents the bash shell from overwriting files, but you can force it with ">|".
export SHELLOPTS # Used to pass above shell options to any called subscripts.

readonly PROGNAME=$(basename $0)
readonly PROGDIR=$(readlink -m $(dirname $0))
readonly ARGS="$@"

num_sps=20
database_dir="database"
wav_dir="${database_dir}/wav"
duration_dir="dur"
htk_label_dir="labels/label_state_align"
mono_label_dir="labels/mono_no_align"
question_dir="questions"
atom_dir="wcad-0.030_0.060_0.090_0.120_0.150"
bap_dir="WORLD/bap"
lf0_dir="WORLD/lf0"
mgc_dir="WORLD/mgc${num_sps}"
vuv_dir="WORLD/vuv"
cmp_dir="WORLD/cmp_mgc${num_sps}"

egs_folder=${1}
file_id_list=${2}
IFS=$'\r\n' GLOBIGNORE='*' command eval 'utts=($(cat $file_id_list))'

# Empty or create fixture directories.
echo "Clear or create all fixture folders."
for dir in ${database_dir} ${wav_dir} ${duration_dir} ${htk_label_dir} ${mono_label_dir} ${question_dir} ${atom_dir} ${bap_dir} ${lf0_dir} ${mgc_dir} ${vuv_dir} ${cmp_dir}; do
    if [ -d "${dir}" ]; then
        echo "    Clear ${dir}"
        rm -fR "${dir}/"*
    else
        echo "    Create ${dir}"
        mkdir -p "${dir}"
    fi
done


echo "Create fixtures for (${#utts[@]}) ${utts[@]}"

# Copy used file_id_list to database folder.
cp ${file_id_list} ${database_dir}/

# Copy required part of utts.data file (contains the textual labels).
pat=$(echo ${utts[@]}|tr " " "|")
grep -Ew "$pat" ${egs_folder}/database/utts.data >| ${database_dir}/utts.data

# Copy more special files.
cp "${egs_folder}/experiments/fixtures/labels/mono_phone.list" "labels/"

# Copy features for utterance ids in given file_id_list to their respective fixture directory.
for id in "${utts[@]}"; do
    # Implementation to loop over tuples.
    OLDIFS=$IFS;
    IFS=',';
    for tuple in "","${wav_dir}"\
                 "experiments/fixtures","${duration_dir}"\
                 "experiments/full","${htk_label_dir}"\
                 "experiments/full/","${mono_label_dir}"\
                 "experiments/fixtures","${question_dir}"\
                 "experiments/fixtures","${atom_dir}"\
                 "experiments/fixtures","${bap_dir}"\
                 "experiments/fixtures","${lf0_dir}"\
                 "experiments/fixtures","${mgc_dir}"\
                 "experiments/fixtures","${vuv_dir}"\
                 "experiments/fixtures","${cmp_dir}"; do
        set -- ${tuple};
        cp "${egs_folder}/${1}/${2}/${id}"* "${2}/"  # $1 is first element of tuple, $2 the second.
    done
    IFS=${OLDIFS}
done

# Copy normalisation parameters (note that not all folder have it).
OLDIFS=$IFS;
IFS=',';
for tuple in "experiments/fixtures","${duration_dir}"\
             "experiments/fixtures","${question_dir}"\
             "experiments/fixtures","${atom_dir}"\
             "experiments/fixtures","${bap_dir}"\
             "experiments/fixtures","${lf0_dir}"\
             "experiments/fixtures","${mgc_dir}"\
             "experiments/fixtures","${cmp_dir}"; do
    set -- ${tuple};
    cp "${egs_folder}/${1}/${2}/"*.bin "${2}/"
done