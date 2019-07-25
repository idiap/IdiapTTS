#!/bin/bash

# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) Sun grid options (IDIAP)
# ATTENTION: Do that in your shell: SETSHELL grid
export cuda_cmd="queue.pl -l gpu"
export cuda_short_cmd="queue.pl -l sgpu"
export cpu_1d_cmd="queue.pl -l q_1day"
export cpu_1d_32G_cmd="queue.pl -l q_1day_mth -pe pe_mth 4" # TODO: Needs testing.
#export cuda_cmd="queue.pl -l q1d,hostname=dynamix03"
#export cuda_cmd="..."
queue_gpu="q_gpu"  # q_gpu, q_short_gpu, cpu
queue_gpu_short="q_short_gpu"
queue_gpu_long="q_long_gpu"
queue_gpu_mth="q_gpu_mth"
queue_cpu_1d="q_1day"
queue_cpu_1d_32G="q_1day_mth -pe pe_mth 4"
queue_cpu_1w="q_1week"
cpu="cpu"

#b) BUT cluster options
#export cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
#export cuda_cmd="queue.pl -q long.q@pcspeech-gpu"

#c) run it locally...
# export cuda_cmd=run.pl
# export cuda_short_cmd=$cuda_cmd
