#!/bin/bash
#
# Copyright 2017 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
# Alexandros Lazaridis
# Pierre-Edouard Honnet
# January 2015
#
# Bastian Schnell
# June 2017
######################################################################

usage() {
cat <<- EOF
    usage: $PROGNAME [OPTIONS] <festival_dir> <input_file> <accent> <output_dir>
    
    Makes HTS labels (full and mono) for all utterances in <input_file> with American (AM) or British (BR) accent.
    Each line in <input_file> should have the format: <utterance_id> <utterance>
    The labes are created at <output_dir>/labels/ in full and mono version.
    <festival_dir> has to point to your festival build.
    <festival_dir> should have a folder with bin/festival executable.

    OPTIONS:
        -h                        show this help

    
    Examples:
        Run the example prompts for an American accent and store the labels in the same directory.
        $PROGNAME ./festival example_English_prompts.txt AM \$PWD

        Run the example prompts for a British accent and store the labels in the same directory.
        $PROGNAME ./festival example_English_prompts.txt BR \$PWD
EOF
}

###############################
# Default options and functions
#
# set -o xtrace # Prints every command before running it, same as "set -x".
# set -o errexit # Exit when a command fails, same as "set -e".
#                # Use "|| true" for those who are allowed to fail.
#                #  Disable (set +e) this mode if you want to know a nonzero return value.
# set -o pipefail # Catch mysqldump fails.
# set -o nounset # Exit when using undeclared variables, same as "set -u".
# set -o noclobber
# Prevents the bash shell from overwriting files, but you can force it with ">|".
export SHELLOPTS # Used to pass above shell options to any called subscripts.

readonly PROGNAME=$(basename $0)
readonly PROGDIR=$(readlink -m $(dirname $0))
readonly ARGS="$@"

# Provide log function.
log()
{
    echo -e >&2 "$@"
}
# Clean up is called by the trap at any stop/exit of the script.
cleanup()
{
    # Cleanup code.
    rm -rf "$TMP"
}
# Die should be called when an error occurs with a HELPFUL error message.
die () {
    log "ERROR" "$@"
    exit 1
}

# Set magic variables for current file $ directory (upper-case).
readonly TMP=$(mktemp -d)

# The main function of this file.
main()
{
    # Force execution of cleanup at the end.
    trap cleanup EXIT INT TERM HUP
    
    log "INFO" "Run ${PROGNAME} $@"

    while getopts ":h" flag; do # If a character is followed by a colon (e.g. f:), that option is expected to have an argument.
        case "${flag}" in
            h) usage; exit ;;
            \?) die "Invalid option: -$OPTARG" ;;
        esac
    done
    shift $(($OPTIND - 1)) # Skip the already processed arguments.

    # Read arguments.
    local expectedArgs=4 # Always use "local" for variables, global variables are evil anyway.
    if [[ $# != "${expectedArgs}" ]]; then
        usage # Function call.
        die "Wrong number of parameters, expected ${expectedArgs} but got $#."
    fi
    
    # Read parameters, use default values (:-) to work with -u option.
    festival_dir=$(realpath "${1:-}")
    local input_file=${2:-}
    local accent=${3:-}
    local output_dir=${4:-}
    # echo $festival_dir

    mkdir -p $output_dir
    lexicon=uni

    # Scripts in $PROGDIR
    local txt2festam=$PROGDIR/Text2FestivalReadyAm.pl
    local utt2labamuni=$PROGDIR/utt2lab-unilex-gam.sh
    local txt2festbr=$PROGDIR/Text2FestivalReadyBr.pl
    local utt2labbruni=$PROGDIR/utt2lab-unilex-rpx.sh

    # Directories
    local utt=$TMP/utt

    # Remove '.txt' in prompt list
    local input_file_nopath=$output_dir/$(basename $input_file).nopath
    sed 's/\.txt//g' $input_file >| $input_file_nopath

    # Prepare list of prompts for festival
    echo -n "Prepare list of prompts for festival ... "
    local input_file_festready=$output_dir/$(basename $input_file).festready
    if [[ $accent == 'AM' ]]; then
        $txt2festam $input_file_nopath $input_file_festready $utt
    elif [[ $accent == 'BR' ]]; then
        $txt2festbr $input_file_nopath $input_file_festready $utt
    else
        echo "Unknown accent: $accent"
        exit 1
    fi
    echo "done."

    # Run festival with this script
    echo "Run festival with this script ..."
    mkdir -p $utt
    $festival_dir/bin/festival $input_file_festready || true
    echo "done."
    
    # Create labels from utts
    echo -n "Create labels from utts ... "
    (
        if [[ $accent == 'AM' ]]; then
	    if [[ $lexicon == 'uni' ]]; then
	        $utt2labamuni $festival_dir $TMP
	    elif [[ $lexicon == 'combi' ]]; then
	        echo "This case isn't treated yet"
	    fi	    
        elif [[ $accent == 'BR' ]]; then
	    if [[ $lexicon == 'uni' ]]; then
	        $utt2labbruni $festival_dir $TMP
	    elif [[ $lexicon == 'combi' ]]; then
	        echo "This case isn't treated yet"
	    fi	    
        else
	    echo "Unknown accent: $accent"
	    exit 1
        fi
    )
    echo "done."
    
    cp -r -f $TMP/labels $output_dir
    echo "Your labels are in $output_dir/labels/full"

    # Cleaning
    rm -rf $input_file_nopath $input_file_festready $utt
}


# Call the main function, provide all parameters.
main "$@"








