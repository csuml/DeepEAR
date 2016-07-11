#!/bin/bash
deepnet=$AIRING_HOME/deepnet/deepnet

prefix=/data1/ningzhang/audio_eating_2/npFeatures/All/LOPO_2_Multimodal_clean

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

# JOINT HIDDEN2 RBM
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_hidden2_joint_rbm_LAST ]; then
	echo "Training Joint hidden2 RBM layer ."
	python merge_dataset_pb.py \
    	${data_output_dir}/${modality1}_rbm2_LAST/data.pbtxt \
    	${data_output_dir}/${modality2}_rbm2_LAST/data.pbtxt \
		${data_output_dir}/${model_prefix}_hidden2_joint_rbm_LAST/input_data.pbtxt || exit 1
	python ${trainer} models/${model_prefix}/hidden2_joint_rbm.pbtxt trainers/${model_prefix}/hidden2_joint/train_hidden2_joint_${MS}.pbtxt eval.pbtxt || exit 1
	python ${extract_rep} ${model_output_dir}/${model_prefix}_hidden2_joint_rbm_LAST trainers/${model_prefix}/hidden2_joint/train_hidden2_joint_CD_${MS}.pbtxt ${model_prefix}_joint_hidden3 \
${data_output_dir}/${model_prefix}_hidden2_joint_rbm_LAST ${gpu_mem} ${main_mem} || exit 1
fi

#train classfier directly 
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_joint_hidden3_classifier_LAST ]; then
echo "train classfier only with representations from joint layer"
	python merge_dataset_pb.py \
    	${prefix}/data_${MS}.pbtxt \
    	${data_output_dir}/${model_prefix}_hidden2_joint_rbm_LAST/data.pbtxt \
    	${data_output_dir}/${model_prefix}_joint_hidden3_classifier/input_data.pbtxt|| exit 1
	python ${trainer} models/${model_prefix}/joint_hidden3_classifier.pbtxt trainers/${model_prefix}/hidden2_joint/train_hidden2_joint_classifier_${MS}.pbtxt eval.pbtxt
fi


# TRAIN HIDDEN2 JOINT DNN
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_hidden2_joint_dnn_BEST ]; then
echo "Training hidden2 joint DNN."
python ${trainer} models/${model_prefix}/dnn_hidden2_joint_${MS}.pbtxt trainers/${model_prefix}/train_dnn_${MS}.pbtxt eval.pbtxt || exit 1
fi

# INPUT JOINT RBM
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_input_joint_rbm_LAST ]; then
    echo "Training input joint RBM layer ."
    python ${trainer} models/${model_prefix}/input_joint_rbm_${MS}.pbtxt trainers/${model_prefix}/input_joint/train_input_joint_${MS}.pbtxt eval.pbtxt || exit 1
    python ${extract_rep} ${model_output_dir}/${model_prefix}_input_joint_rbm_LAST trainers/${model_prefix}/input_joint/train_input_joint_CD_${MS}.pbtxt ${model_prefix}_joint_hidden1 \
${data_output_dir}/${model_prefix}_input_joint_rbm_LAST ${gpu_mem} ${main_mem} || exit 1
fi

#JOINT HIDDEN1 CLASSIFIER
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_joint_hidden1_classifier_BEST ]; then
echo "train classfier only with representations from ${model_prefix}_joint_hidden1"
python merge_dataset_pb.py \
    ${prefix}/data_${MS}.pbtxt \
    ${data_output_dir}/${model_prefix}_input_joint_rbm_LAST/data.pbtxt \
    ${data_output_dir}/${model_prefix}_joint_hidden1_classifier_LAST/input_data.pbtxt|| exit 1
python ${trainer} models/${model_prefix}/joint_hidden1_classifier.pbtxt trainers/${model_prefix}/input_joint/train_joint_hidden1_classifier_${MS}.pbtxt eval.pbtxt
fi


#LAYER 2 
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_rbm2_LAST ]; then
    echo "Training  RBM2 layer ."
    python ${trainer} models/${model_prefix}/joint_rbm2.pbtxt trainers/${model_prefix}/train_normal_rbm2_${MS}.pbtxt eval.pbtxt || exit 1
    python ${extract_rep} ${model_output_dir}/${model_prefix}_rbm2_LAST trainers/${model_prefix}/train_normal_rbm2_${MS}.pbtxt ${model_prefix}_hidden2 \
${data_output_dir}/${model_prefix}_rbm2_LAST ${gpu_mem} ${main_mem} || exit 1
fi

#LAYER2 CLASSIFIER
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_hidden2_classifier_BEST ]; then
echo "train classfier only with representations from ${model_prefix}_hidden2"
python merge_dataset_pb.py \
    ${prefix}/data_${MS}.pbtxt \
    ${data_output_dir}/${model_prefix}_rbm2_LAST/data.pbtxt \
    ${data_output_dir}/${model_prefix}_hidden2_classifier_LAST/input_data.pbtxt|| exit 1
python ${trainer} models/${model_prefix}/hidden2_normal_classifier.pbtxt trainers/${model_prefix}/train_hidden2_normal_classifier_${MS}.pbtxt eval.pbtxt
fi

#LAYER 3 
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_rbm3_LAST ]; then
    echo "Training input RBM3 layer ."
    python ${trainer} models/${model_prefix}/joint_rbm3.pbtxt trainers/${model_prefix}/train_normal_rbm3_${MS}.pbtxt eval.pbtxt || exit 1
    python ${extract_rep} ${model_output_dir}/${model_prefix}_rbm3_LAST trainers/${model_prefix}/train_normal_rbm3_${MS}.pbtxt ${model_prefix}_hidden3 \
${data_output_dir}/${model_prefix}_rbm3_LAST ${gpu_mem} ${main_mem} || exit 1
fi

#LAYER3 CLASSIFIER
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_hidden3_classifier_BEST ]; then
echo "train classfier only with representations from ${model_prefix}_hidden3"
python merge_dataset_pb.py \
    ${prefix}/data_${MS}.pbtxt \
    ${data_output_dir}/${model_prefix}_rbm3_LAST/data.pbtxt \
    ${data_output_dir}/${model_prefix}_hidden3_classifier_LAST/input_data.pbtxt|| exit 1
python ${trainer} models/${model_prefix}/hidden3_normal_classifier.pbtxt trainers/${model_prefix}/train_hidden3_normal_classifier_${MS}.pbtxt eval.pbtxt
fi

# TRAIN DNN
if ${clobber} || [ ! -e ${model_output_dir}/${model_prefix}_input_joint_dnn_BEST ]; then
echo "Training input joint DNN."
python ${trainer} models/${model_prefix}/input_joint_dnn_${MS}.pbtxt trainers/${model_prefix}/train_dnn_${MS}.pbtxt eval.pbtxt || exit 1
fi
