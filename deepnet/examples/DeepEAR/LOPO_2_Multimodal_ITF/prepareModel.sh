#!/bin/bash

cd models
for feature in ZCR MFCC Spectral Chroma Energy PLP
do
    echo "preparing model files for ${feature}"
    cd ${feature}
    #sed -i "s/input_layer/${feature}_input_layer/" *
	#sed -i "s/hidden/${feature}_hidden/" *
	sed -i "s/${feature}_${feature}/${feature}/" *
    cd ..
done
