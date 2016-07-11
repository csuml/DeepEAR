#!/etc/bash

for feature in ZCR MFCC Spectral Chroma Energy PLP
do 
	echo "processing ${feature}"
	sh runall.sh $feature withMS
	#sh runall.sh $feature withoutMS
done 
