'''
Created on Jun 20, 2016

@author: airingzhang
'''
import os
import glob as glob
import numpy as np
import sys
from deepnet import util
from google.protobuf import text_format
from deepnet import deepnet_pb2

def normal(path1, path2, path3, readPbtxt = False):
    datasets = ["train", "validation", "test"]
    for dataset in datasets:
        fileNames1 = sorted(glob.glob(os.path.join(path1, dataset, "*.npy")))
        fileNames2 = sorted(glob.glob(os.path.join(path2, dataset, "*.npy")))
        for i, file1, file2 in enumerate(zip(fileNames1, fileNames2)):
            data1 = np.load(file1)
            data2 = np.load(file2)
            dataCombined = np.concatenate((data1, data2), axis = 1) 
            if i == 0:   
                data = dataCombined   
            else:
                data = np.concatenate((data, dataCombined), axis = 0)
        np.save(os.path.join(path3, dataset, "data"), data)
        
def withPbtxt(dbPbtxt, modality1, modality2, outputpath):  
    datapb = util.ReadData(dbPbtxt)
    datasets = ["train", "validation", "test"]
    datapbNew = deepnet_pb2.Dataset()  
    namePrefix = modality1 + "_"+ modality2 +"_"
    datapbNew.prefix = outputpath
    datapbNew.name = namePrefix + "combined_input"
    for dataset in datasets:
        fileNames1 = []
        fileNames2 = []
        for dataEntry in datapb.data:
            if modality1 in dataEntry.name and dataset in dataEntry.name:
                fileNames1 = sorted(glob.glob(os.path.join(datapb.prefix, dataEntry.file_pattern)))
            if modality2 in dataEntry.name and dataset in dataEntry.name:
                fileNames2 = sorted(glob.glob(os.path.join(datapb.prefix, dataEntry.file_pattern)))
        for i, (file1, file2) in enumerate(zip(fileNames1, fileNames2)):
            data1 = np.load(file1)
            data2 = np.load(file2)
            dataCombined = np.concatenate((data1, data2), axis = 1) 
            if i == 0:   
                data = dataCombined   
            else:
                data = np.concatenate((data, dataCombined), axis = 0)
        if not os.path.exists(os.path.join(outputpath, dataset)):
            os.makedirs(os.path.join(outputpath, dataset))
        np.save(os.path.join(outputpath, dataset, "data"), data)    
        dataItem = deepnet_pb2.Dataset.Data()
        dataItem.name = namePrefix + "combined_" + dataset
        dataItem.dimensions.extend([data.shape[1]]) 
        dataItem.size = data.shape[0]
        dataItem.file_pattern = os.path.join(dataset,"*.npy")
        datapbNew.data.extend([dataItem]) 
    with open(os.path.join(outputpath, "input_data.pbtxt"), 'w') as f:
        text_format.PrintMessage(datapbNew, f) 
          
if __name__ == '__main__':
    readPbtxt = False
    if sys.argv[4].upper() == "TRUE":
        readPbtxt = True
        normal(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        withPbtxt(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        
