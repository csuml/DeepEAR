#!/bin/bash
deepnet=$AIRING_HOME/deepnet/deepnet

prefix=/data1/ningzhang/audio_eating_2/Data/LOPO_1

# Amount of gpu memory to be used for buffering data. Adjust this for your GPU.
# For a GPU with 6GB memory, this should be around 4GB.
# If you get 'out of memory' errors, try decreasing this.
gpu_mem=4G

# Amount of main memory to be used for buffering data. Adjust this according to
# your RAM. Having atleast 16G is ideal.
main_mem=20G

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${prefix}/models
data_output_dir=${prefix}/reps
clobber=false

python ${deepnet}/online_test.py ${model_output_dir}/dnn_LAST trainers/train_dnn.pbtxt eval.pbtxt ${prefix}/test/environment_25.npy # ${prefix}/test/P256_eating_t3.npy

