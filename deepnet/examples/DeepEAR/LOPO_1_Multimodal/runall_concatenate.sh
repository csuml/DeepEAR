#!/bin/bash
deepnet=$AIRING_HOME/deepnet/deepnet

prefix=/data1/ningzhang/audio_eating_2/npFeatures/All/LOPO_1_Multimodal

# Amount of gpu memory to be used for buffering data. Adjust this for your GPU.
# For a GPU with 6GB memory, this should be around 4GB.
# If you get 'out of memory' errors, try decreasing this.
gpu_mem=4G

# Amount of main memory to be used for buffering data. Adjust this according to
# your RAM. Having atleast 16G is ideal.
main_mem=20G

modality1=$1
modality2=$2
MS=$3
trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${prefix}/models/${MS}
data_output_dir=${prefix}/reps/${MS}
clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

model_prefix=${modality1}_${modality2}
if [ ! -e models/${model_prefix} ]; then
	model_prefix=${modality2}_${modality1}
	if [ ! -e models/${model_prefix} ]; then
	echo "model definition to combin ${modality1} and ${modality2} does not exist"
	fi
fi

if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_combined_rbm1_LAST ]; then 
	echo "Training ${model_prefix}_combined_rbm1"
	if [ ! -e ${data_output_dir}/${model_prefix}_combined_rbm1_LAST/input_data.pbtxt ]; then 
		echo "preparing input data"
		python concatenateData.py ${prefix}/data_${MS}.pbtxt ${modality1} ${modality2} ${data_output_dir}/${model_prefix}_combined_rbm1_LAST/
	fi  
	python ${trainer} models/${model_prefix}/rbm1_${MS}.pbtxt trainers/${model_prefix}/train_rbm1_${MS}.pbtxt eval.pbtxt || exit 1
	python ${extract_rep} ${model_output_dir}/${model_prefix}_combined_rbm1_LAST trainers/${model_prefix}/train_rbm1_${MS}.pbtxt ${model_prefix}_combined_hidden1 \
	${data_output_dir}/${model_prefix}_combined_rbm1_LAST ${gpu_mem} ${main_mem} || exit 1
fi


if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_combined_rbm2_LAST ]; then
    echo "Training ${model_prefix}_combined_rbm2"
    python ${trainer} models/${model_prefix}/rbm2.pbtxt trainers/${model_prefix}/train_rbm2_${MS}.pbtxt eval.pbtxt || exit 1
    python ${extract_rep} ${model_output_dir}/${model_prefix}_combined_rbm2_LAST trainers/${model_prefix}/train_rbm2_${MS}.pbtxt ${model_prefix}_combined_hidden2 \
${data_output_dir}/${model_prefix}_combined_rbm2_LAST ${gpu_mem} ${main_mem} || exit 1
fi

if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_combined_rbm3_LAST ]; then
    echo "Training ${model_prefix}_combined_rbm3"
    python ${trainer} models/${model_prefix}/rbm3.pbtxt trainers/${model_prefix}/train_rbm3_${MS}.pbtxt eval.pbtxt || exit 1
    python ${extract_rep} ${model_output_dir}/${model_prefix}_combined_rbm3_LAST trainers/${model_prefix}/train_rbm3_${MS}.pbtxt ${model_prefix}_combined_hidden3 \
${data_output_dir}/${model_prefix}_combined_rbm3_LAST ${gpu_mem} ${main_mem} || exit 1
fi

if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_combined_dnn_LAST ]; then
    echo "Training ${model_prefix}_combined_dnn"
    python merge_dataset_pb.py \
            ${prefix}/data_${MS}.pbtxt \
            ${data_output_dir}/${model_prefix}_combined_rbm1_LAST/input_data.pbtxt \
            ${data_output_dir}/${model_prefix}_combined_rbm1_LAST/input_data.pbtxt || exit 1
    python ${trainer} models/${model_prefix}/dnn_combined_${MS}.pbtxt trainers/${model_prefix}/train_dnn_combined_${MS}.pbtxt eval.pbtxt || exit 1
fi

# COMBINED HIDDEN CLASSIFIER
for index in `seq 3`
do 
(
	if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_combined_hidden${index}_classifier_LAST ]; then
    	echo "Training hidden${index} classifer ."
    	python merge_dataset_pb.py \
          	 ${data_output_dir}/${model_prefix}_combined_rbm${index}_LAST/data.pbtxt \
           	 ${prefix}/data_${MS}.pbtxt \
        	 ${data_output_dir}/${model_prefix}_combined_hidden${index}_classifier_LAST/input_data.pbtxt
    	python ${trainer} models/${model_prefix}/hidden${index}_classifier.pbtxt trainers/${model_prefix}/train_hidden${index}_classifier_${MS}.pbtxt eval.pbtxt || exit 1
	fi
)
done

if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_combined_input_classifier_LAST ]; then
   echo "Training combined input classifer ."
   python ${trainer} models/${model_prefix}/combined_input_classifier_${MS}.pbtxt trainers/${model_prefix}/train_input_classifier_${MS}.pbtxt eval.pbtxt || exit 1
fi

