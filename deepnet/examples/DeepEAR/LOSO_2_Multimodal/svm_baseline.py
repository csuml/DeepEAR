'''
Created on Jun 24, 2016

@author: airingzhang
'''
import numpy as np
import glob as glob
import sys
import os
from deepnet import deepnet_pb2
from google.protobuf import text_format
from svmutil import * 
from deepnet import util
import pdb

def convertToTxt(data, label, path):
    o = open( path, 'wb' )
    #pdb.set_trace()
    rows, columns = data.shape
    for i in range(rows):
        new_line = []
        tag = "1" if float(label[i]) == 1.0 else "-1"
        new_line.append( tag )
        for j in range(columns):
            if data[i,j] == '' or float( data[i,j] ) == 0.0:
                continue
            new_item = "%s:%s" % ( j + 1, data[i,j] )
            new_line.append( new_item )
        new_line = " ".join( new_line )
        new_line += "\n"
        o.write(new_line)
        
def fromPb(datapb, modality, outputPath, saverawData = False, skipTraining = False):
    datasets = ["train", "validation", "test",]
    dataMap = {}
    #newdataPb = deepnet_pb2.Dataset()
    #newdataPb.name = datapb.name + "_comebined"
    #newdataPb.prefix = datapb.prefix
    for dataset in datasets:
        dataMap[dataset] = {}
    pdb.set_trace()
    for entry in datapb.data:
        for dataset in datasets:
            
            if dataset in entry.name:
                if "label" in entry.name:
                    rawList = []
                    filenames = sorted(glob.glob(os.path.join(datapb.prefix,entry.file_pattern))) 
                    for i, filename in enumerate(filenames):
                        rawList.append(np.load(filename))
                    combined = np.concatenate(tuple(rawList), axis = 0)
                    dataMap[dataset]["label"] = combined *2.0 -1.0
                    if saverawData:
                        dir, base = os.path.split(filename)
                        dir, base = os.path.split(dir)
                        baseName =  "label_" + dataset
                        np.save(os.path.join(dir, baseName), combined)
                if modality in entry.name:
                    rawList = []
                    filenames = sorted(glob.glob(os.path.join(datapb.prefix,entry.file_pattern)))
                    for i, filename in enumerate(filenames):
                        rawList.append(np.load(filename))
                    combined = np.concatenate(tuple(rawList), axis = 0)
                    dataMap[dataset]["data"] = combined
                    if saverawData:
                        dir, base = os.path.split(filename)
                        dir, base = os.path.split(dir)
                        baseName = modality + "_" + dataset
                        np.save(os.path.join(dir, baseName), combined)
    if saverawData:
        for dataset in datasets:
            convertToTxt(dataMap[dataset]["data"], dataMap[dataset]["label"], os.path.join(dir, modality+"_libsvm_"+dataset))
              
    if skipTraining:
        for dataset in datasets:
            dataMap[dataset]["label"] = dataMap[dataset]["label"].tolist()
            dataMap[dataset]["data"] = dataMap[dataset]["data"].tolist()
            
        prob = svm_problem(dataMap["train"]["label"], dataMap["train"]["data"])
        param = svm_parameter('-t 2 -c 4 -b 1')
        m = svm_train(prob, param)
        svm_save_model(os.path.join(outputPath,modality+'_svm.model'), m)
        p_label, p_acc, p_val = svm_predict(dataMap["validation"]["label"], dataMap["validation"]["data"] , m, '-b 1')
        ACC, MSE, SCC = evaluations(dataMap["validation"]["label"], p_label)
        print "ACC on validation set: " + repr(ACC) 
        p_label, p_acc, p_val = svm_predict(dataMap["test"]["label"], dataMap["test"]["data"] , m, '-b 1')
        ACC, MSE, SCC = evaluations(dataMap["test"]["label"], p_label)
        print "ACC on test set: " + repr(ACC) 


if __name__ == '__main__':
    pbPath = sys.argv[1]
    modality = sys.argv[2]
    outputPath = sys.argv[3]
    if len(sys.argv) > 4:
        saveFile = True if sys.argv[4].upper() == "TRUE" else False
    if len(sys.argv) > 5:
        skipTraining = True if sys.argv[5].upper() == "TRUE" else False
    datapb = util.ReadData(pbPath)
    fromPb(datapb, modality, outputPath, saveFile, skipTraining)
