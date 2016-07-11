#!/bin/bash
deepnet=$AIRING_HOME/deepnet/deepnet

prefix=/data1/ningzhang/audio_eating_2/npFeatures/All/LOPO_4_Multimodal_ITF

# Amount of gpu memory to be used for buffering data. Adjust this for your GPU.
# For a GPU with 6GB memory, this should be around 4GB.
# If you get 'out of memory' errors, try decreasing this.
gpu_mem=4G

# Amount of main memory to be used for buffering data. Adjust this according to
# your RAM. Having atleast 16G is ideal.
main_mem=20G

feature=$1
MS=$2
trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${prefix}/models/${MS}
data_output_dir=${prefix}/reps/${MS}
clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

#TRAIN INPUT CLASSIFIER
if ${clobber} || [ ! -e ${model_output_dir}/${feature}_input_classifier_LAST ]; then
    echo "Training input classifier ."
    python ${trainer} models/${feature}/input_classifier_${MS}.pbtxt trainers/${feature}/train_input_classifier_${MS}.pbtxt eval.pbtxt || exit 1
fi

# LAYER - 1.
#(
if ${clobber} || [ ! -e ${model_output_dir}/${feature}_rbm1_LAST ]; then
echo "Training first RBM layer ."
python ${trainer} models/${feature}/rbm1_${MS}.pbtxt trainers/${feature}/train_rbm1_${MS}.pbtxt eval.pbtxt || exit 1
python ${extract_rep} ${model_output_dir}/${feature}_rbm1_LAST trainers/${feature}/train_rbm1_${MS}.pbtxt ${feature}_hidden1 \
${data_output_dir}/${feature}_rbm1_LAST ${gpu_mem} ${main_mem} || exit 1
cp ${data_output_dir}/${feature}_rbm1_LAST/data.pbtxt ${data_output_dir}/${feature}_rbm1_LAST/data_withID.pbtxt
sed -i "s/\"hidden/\"${feature}_hidden/" ${data_output_dir}/${feature}_rbm1_LAST/data_withID.pbtxt
fi

#HIDDEN1 CLASSIFIER
if ${clobber} || [ ! -e ${model_output_dir}/${feature}_hidden1_classifier_LAST ]; then
    echo "Training hidden1 classifier ."
    python merge_dataset_pb.py \
           ${data_output_dir}/${feature}_rbm1_LAST/data.pbtxt \
           ${prefix}/data_${MS}.pbtxt \
           ${data_output_dir}/${feature}_hidden1_classifier_LAST/input_data.pbtxt
    python ${trainer} models/${feature}/hidden1_classifier.pbtxt trainers/${feature}/train_hidden1_classifier_${MS}.pbtxt eval.pbtxt || exit 1
fi


# IMAGE LAYER - 2.
if ${clobber} || [ ! -e ${model_output_dir}/${feature}_rbm2_LAST ]; then
echo "Training second RBM layer ."
python ${trainer} models/${feature}/rbm2.pbtxt trainers/${feature}/train_rbm2_${MS}.pbtxt eval.pbtxt || exit 1
python ${extract_rep} ${model_output_dir}/${feature}_rbm2_LAST trainers/${feature}/train_rbm2_${MS}.pbtxt ${feature}_hidden2 \
${data_output_dir}/${feature}_rbm2_LAST ${gpu_mem} ${main_mem} || exit 1
cp ${data_output_dir}/${feature}_rbm2_LAST/data.pbtxt ${data_output_dir}/${feature}_rbm2_LAST/data_withID.pbtxt
sed -i "s/\"hidden/\"${feature}_hidden/" ${data_output_dir}/${feature}_rbm2_LAST/data_withID.pbtxt
fi

#HIDDEN2 CLASSIFIER
if ${clobber} || [ ! -e ${model_output_dir}/${feature}_hidden2_classifier_LAST ]; then
    echo "Training hidden2 classifier ."
    python merge_dataset_pb.py \
           ${data_output_dir}/${feature}_rbm2_LAST/data.pbtxt \
           ${prefix}/data_${MS}.pbtxt \
           ${data_output_dir}/${feature}_hidden2_classifier_LAST/input_data.pbtxt
    python ${trainer} models/${feature}/hidden2_classifier.pbtxt trainers/${feature}/train_hidden2_classifier_${MS}.pbtxt eval.pbtxt || exit 1
fi


# IMAGE LAYER - 3.
if ${clobber} || [ ! -e ${model_output_dir}/${feature}_rbm3_LAST ]; then
echo "Training third RBM layer ."
python ${trainer} models/${feature}/rbm3.pbtxt trainers/${feature}/train_rbm3_${MS}.pbtxt eval.pbtxt || exit 1
python ${extract_rep} ${model_output_dir}/${feature}_rbm3_LAST trainers/${feature}/train_rbm3_${MS}.pbtxt ${feature}_hidden3 \
${data_output_dir}/${feature}_rbm3_LAST ${gpu_mem} ${main_mem} || exit 1
cp ${data_output_dir}/${feature}_rbm3_LAST/data.pbtxt ${data_output_dir}/${feature}_rbm3_LAST/data_withID.pbtxt
sed -i "s/\"hidden/\"${feature}_hidden/" ${data_output_dir}/${feature}_rbm3_LAST/data_withID.pbtxt
fi

if ${clobber} || [ ! -e ${model_output_dir}/${feature}_hidden3_classifier_LAST ]; then
    echo "Training hidden3 classifier ."
    python merge_dataset_pb.py \
           ${data_output_dir}/${feature}_rbm3_LAST/data.pbtxt \
           ${prefix}/data_${MS}.pbtxt \
           ${data_output_dir}/${feature}_hidden3_classifier_LAST/input_data.pbtxt
    python ${trainer} models/${feature}/hidden3_classifier.pbtxt trainers/${feature}/train_hidden3_classifier_${MS}.pbtxt eval.pbtxt || exit 1
fi

# TRAIN DNN
if ${clobber} || [ ! -e ${model_output_dir}/${feature}_dnn_BEST ]; then
echo "Training DNN."
python ${trainer} models/${feature}/dnn_${MS}.pbtxt trainers/${feature}/train_dnn_${MS}.pbtxt eval.pbtxt || exit 1
fi

#wait;
# COLLECT RESULTS AND PUT INTO A LATEX TABLE.
#if ${clobber} || [ ! -e results.tex ]; then
#python create_results_table.py ${prefix}/models results.tex || exit 1
#fi

