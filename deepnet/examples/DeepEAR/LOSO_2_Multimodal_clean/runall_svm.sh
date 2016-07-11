#!/bin/bash
deepnet=$AIRING_HOME/deepnet/deepnet

prefix=/data1/ningzhang/audio_eating_2/npFeatures/All/LOSO_2_Multimodal_clean

# Amount of gpu memory to be used for buffering data. Adjust this for your GPU.
# For a GPU with 6GB memory, this should be around 4GB.
# If you get 'out of memory' errors, try decreasing this.
gpu_mem=4G

# Amount of main memory to be used for buffering data. Adjust this according to
# your RAM. Having atleast 16G is ideal.
main_mem=20G

MS=$1
trainer=${deepnet}/trainer.py
model_output_dir=${prefix}/models/${MS}
data_output_dir=${prefix}/reps/${MS}
clobber=false
svmgpu=~/source-linux-v1.2/svm-train-gpu-2

mkdir ${prefix}/onefile

#Prepare libsvm file 
if ${clobber} || [ ! -e ${prefix}/onefile/MFCC_libsvm_train ]; then
	echo "prepare MFCC libsvm file"
	python svm_baseline.py ${prefix}/data_withMS.pbtxt MFCC ${prefix}/onefile true false
fi
if ${clobber} || [ ! -e ${prefix}/onefile/Energy_libsvm_train ]; then
    echo "prepare Energy libsvm file"
	python svm_baseline.py ${prefix}/data_withMS.pbtxt Energy ${prefix}/onefile true false
fi  
if ${clobber} || [ ! -e ${prefix}/onefile/MFCC_Energy_combined_libsvm_train ]; then
	echo "prepare MFCC_Energy libsvm file"
	python svm_baseline.py ${data_output_dir}/MFCC_Energy_combined_rbm1_LAST/input_data.pbtxt MFCC_Energy_combined ${prefix}/onefile true false
fi
 
if ${clobber} || [ ! -e ${prefix}/onefile/MFCC_Energy_combined_libsvm_train ]; then
mv ${prefix}/*/*libsvm* ${prefix}/onefile
mv ${data_output_dir}/*/*libsvm* ${prefix}/onefile
fi

if ${clobber} || [ ! -e ${prefix}/onefile/MFCC_svm_gpu_2.model ]; then
    echo "Training input MFCC svm."
	${svmgpu} -t 2 -c 4  ${prefix}/onefile/MFCC_libsvm_train ${prefix}/onefile/MFCC_svm_gpu_2.model
fi

if ${clobber} || [ ! -e ${prefix}/onefile/Energy_svm_gpu_2.model ]; then
    echo "Training input Energy svm."
    ${svmgpu} -t 2 -c 4  ${prefix}/onefile/Energy_libsvm_train ${prefix}/onefile/Energy_svm_gpu_2.model
fi

if ${clobber} || [ ! -e ${prefix}/onefile/MFCC_Energy_combined_svm_gpu_2.model ]; then
    echo "Training input MFCC_Energy combined svm."
    ${svmgpu} -t 2 -c 4  ${prefix}/onefile/MFCC_Energy_combined_libsvm_train ${prefix}/onefile/MFCC_Energy_combined_svm_gpu_2.model
fi

echo "Testing input MFCC svm with validation set "
${svmgpu} -t 2 -c 4 -v 2 ${prefix}/onefile/MFCC_libsvm_validation ${prefix}/onefile/MFCC_svm_gpu_2.model ${prefix}/onefile/MFCC_libsvm_train

echo "Testing input MFCC svm with test set "
${svmgpu} -t 2 -c 4 -v 2 ${prefix}/onefile/MFCC_libsvm_test ${prefix}/onefile/MFCC_svm_gpu_2.model ${prefix}/onefile/MFCC_libsvm_train

echo "Testing input Energy svm with validation set "
${svmgpu} -t 2 -c 4 -v 2 ${prefix}/onefile/Energy_libsvm_validation ${prefix}/onefile/Energy_svm_gpu_2.model ${prefix}/onefile/Energy_libsvm_train

echo "Testing input Energy svm with test set "
${svmgpu} -t 2 -c 4 -v 2 ${prefix}/onefile/Energy_libsvm_test ${prefix}/onefile/Energy_svm_gpu_2.model ${prefix}/onefile/Energy_libsvm_train

echo "Testing input MFCC_Energy_combined svm with validation set "
${svmgpu} -t 2 -c 4 -v 2 ${prefix}/onefile/MFCC_Energy_combined_libsvm_validation ${prefix}/onefile/MFCC_Energy_combined_svm_gpu_2.model ${prefix}/onefile/MFCC_Energy_combined_libsvm_train

echo "Testing input MFCC_Energy_combined svm with test set "
${svmgpu} -t 2 -c 4 -v 2 ${prefix}/onefile/MFCC_Energy_combined_libsvm_test ${prefix}/onefile/MFCC_Energy_combined_svm_gpu_2.model ${prefix}/onefile/MFCC_Energy_combined_libsvm_train


