#!/etc/bash

for feature in MFCC Energy 
do 
	echo "processing ${feature}"
	sh runall.sh $feature withMS
	#sh runall.sh $feature withoutMS
done 
