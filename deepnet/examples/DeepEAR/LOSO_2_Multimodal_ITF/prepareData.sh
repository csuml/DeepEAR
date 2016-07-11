#!/bin/bash

baseDir=/data1/ningzhang/audio_eating_2/npFeatures/All/LOSO_2_Multimodal_ITF
MS=$1

if [ ${MS} = "withoutMS"  ]; then
	echo "preparing features without MS"
	withMS=false
else
	echo "preparing features with MS"
	withMS=true
fi

for folder in train validation test
do
#(
if [ ! -e ${baseDir}/${folder}/${MS}/data_${MS}.pbtxt ]; then
	echo processing ${folder}
	mkdir -p ${baseDir}/${folder}/${MS}
	python features_parser.py $baseDir/${folder} ${withMS} false || exit 1
fi
#)&
done
#wait
echo "mergeall datasets into one pbtxt file"
python merge_dataset_pb.py ${baseDir}/train/${MS}/data_${MS}.pbtxt \
	${baseDir}/validation/${MS}/data_${MS}.pbtxt \
	${baseDir}/data_${MS}.pbtxt

python merge_dataset_pb.py ${baseDir}/test/${MS}/data_${MS}.pbtxt \
    ${baseDir}/data_${MS}.pbtxt ${baseDir}/data_${MS}.pbtxt


