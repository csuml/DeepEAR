Contact Information:
nzhang@cs.uml.edu


One can go into any sub-folders in this folder ex. LOSO_3_Multimodal_ITF
To run the experiments

So many sub-folders exist because we run multimodal testing case. Basically code in each sub-folders are the same.
Difference only lies in the configurations about the path where the intermediate results are saved.

1. create experiment folder to store intermediate results <LOSO_3_Multimodal_ITF>, and modify all related path in each *sh and *pbtxt 

2. create train, validation, test folder to store the raw features and put relative dataset partitions in each folder respectively

3. parse the raw features into unimodal features: sh prepareData.sh <withMS or withMS>
-withMS means includes the Mean and STD components
-withoutMS means excludes the Mean and STD components

4. run unimodal DBM-DNN experiments: sh batchRun.sh <FeatureName> <withMS or withoutMS>
-FeatureName canbe following options: MFCC Energy Spectral Chroma PLP ZCR
-withMS means includes the Mean and STD components 
-withoutMS means excludes the Mean and STD components 

5. run concatenated DBM-DNN experiment sh: runall_concatenated.sh MFCC Energy withMS

6. run fusion DBM-DNN experiment sh: runall_multimodal.sh MFCC Energy withMS

7. to collect all results one can run scripts below
python create_results_multimodal.py <folder where Models saved > <file_name>
result will be written into the <file_name>


