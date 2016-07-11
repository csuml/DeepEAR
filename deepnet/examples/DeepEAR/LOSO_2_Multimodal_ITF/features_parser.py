'''
Created on Jun 6, 2016

@author: airingzhang
'''
import numpy as np
import glob as glob
import sys
import os
from deepnet import deepnet_pb2
from google.protobuf import text_format

class FeatureParser (object):
    '''
    This file is used for audio project with Prof. Chen to parse the concatenated Features.
    Also it provide method to group features
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.featureGroups = ['ZCR', 'Energy', 'Spectral', 'Chroma', 'PLP', 'MFCC']
        self.featureGroupsIndex = [0, 401, 1203, 2406, 7218, 12405, 17592]
        self.featureGroupsDict = {'ZCR':[(0,399)], 'Energy':[(401, 800), (802,1201)], 
                                  'Spectral':[(1203,1602),(1604,2003),(2005, 2404)], 'Chroma': [(2406,7194)], 
                                  'PLP':[(7218,12405)], 'MFCC':[(12405,17592)]}
        self.subPersonDir = ['P1', 'P1222', 'P176', 'P201', 'P221', 'P241', 'P252', 'P255', 'P3',
                       'P331', 'P5', 'P599', 'P601', 'P8', 'P9', 'P1001', 'P169', 'P2', 'P21',
                       'P231', 'P251', 'P253', 'P256', 'P330', 'P4', 'P50', 'P6', 'P7', 'P86']
        
        self.subTypeDir = ['Type1', 'Type2', 'Type3', 'Type4']
        self.subOtherDir = ['env', 's']
        
    def ParsePerson(self, baseDir, ne=True, withMS = False):
        self.baseDir = baseDir
        for person in self.subPersonDir:
            instanceCount = 0
            dataPb = deepnet_pb2.Dataset()
            outputProtoFile = os.path.join(self.baseDir, person,'data.pbtxt')
            for i, feature in enumerate(self.featureGroups):
                data = deepnet_pb2.Dataset.Data()
                data.name = person+"_"+feature
                data.file_pattern = "*"+feature+".npy"
                if withMS:
                    data.dimensions.extend([self.featureGroupsIndex[i+1]-self.featureGroupsIndex[i]])
                else:
                    dimensions = 0
                    for entry in self.featureGroupsDict[feature]:
                        dimensions = dimensions + entry[1] - entry[0]
                data.dimensions.extend([dimensions])
                dataPb.data.extend([data]) 
                
            data = deepnet_pb2.Dataset.Data()
            data.name = person+"_label"
            data.dimensions.extend([1]) 
            data.file_pattern = "*label.npy"
            dataPb.data.extend([data]) 
            dataPb.prefix = os.path.join(self.baseDir, person) 
            if withMS:
                dataPb.name = os.path.basename(baseDir) + "withMS"
                outputProtoFile = os.path.join(baseDir, 'data_withMS.pbtxt')
            else:
                dataPb.name = os.path.basename(baseDir) + "withoutMS"
                outputProtoFile = os.path.join(baseDir, 'data_withoutMS.pbtxt')   
            if ne:
                filePath = os.path.join(self.baseDir, person, "*.npy")
                files = glob.glob(filePath)
                for fileEntry in files:
                    tempData = np.load(fileEntry)
                    if tempData.shape[1] == 17593:
                        continue
                    instanceCount = instanceCount + tempData.shape[0]
                    
                    fileName = os.path.splitext(fileEntry)[0]
                    if withMS:
                        for i, feature in self.featureGroups:
                            np.save(fileName + '_' + feature + "_withMS.npy", tempData[:, self.featureGroupsIndex[i]:self.featureGroupsIndex[i + 1]])
                    else:
                        for feature in self.featureGroups:
                            tempTuple = self.featureGroupsDict[feature][0]
                            tempArray = tempData[:, tempTuple[0]: tempTuple[1]]
                            if len(self.featureGroupsDict[feature]) > 1:
                                for i in range(1, len(self.featureGroupsDict[feature])):
                                    tempTuple = self.featureGroupsDict[feature][i]
                                    tempArray = np.concatenate((tempArray, tempData[:,tempTuple[0]: tempTuple[1]]), axis = 1)
                            np.save(fileName + '_' + feature + "_withoutMS.npy", tempArray)    
                    np.save(fileName + '_label.npy', tempData[:, 17592])
                                        
            else:  
                for fType in self.subTypeDir:
                    filePath = os.path.join(self.baseDir, person, fType, "*.npy")
                    files = glob.glob(filePath)
                    for fileEntry in files:
                        tempData = np.load(fileEntry)
                        assert(tempData.shape[1] == 17593)
                        instanceCount = instanceCount + tempData.shape[0]
                        
                        baseName = os.path.splitext(os.path.basename(fileEntry))[0]
                        fileName = os.path.join(self.baseDir, person, baseName)
                        if withMS:
                            for i, feature in enumerate(self.featureGroups):
                                np.save(fileName + '_' + feature + "_withtMS.npy", tempData[:, self.featureGroupsIndex[i]:self.featureGroupsIndex[i + 1]])
                            
                        else:
                            for feature in self.featureGroups:
                                tempTuple = self.featureGroupsDict[feature][0]
                                tempArray = tempData[:, tempTuple[0]: tempTuple[1]]
                                if len(self.featureGroupsDict[feature]) > 1:
                                    for i in range(1, len(self.featureGroupsDict[feature])):
                                        tempTuple = self.featureGroupsDict[feature][i]
                                        tempArray = np.concatenate((tempArray, tempData[:,tempTuple[0]: tempTuple[1]]), axis = 1)
                                np.save(fileName + '_' + feature + "_withoutMS.npy", tempArray) 
                        np.save(fileName + '_label.npy', tempData[:, 17592])        
            for entry in dataPb.data:
                entry.size = instanceCount
            with open(outputProtoFile, 'w') as f:
                text_format.PrintMessage(dataPb, f) 
                
    def ParseOther(self, baseDir, withMS = False):
        self.baseDir = baseDir
        pathDir = os.path.join(baseDir, "*.npy")
        files = glob.glob(pathDir)
        instanceCount = 0
        dataPb = deepnet_pb2.Dataset()
        
        for i, feature in enumerate(self.featureGroups):
            data = deepnet_pb2.Dataset.Data()
            data.name = feature + "_"+ os.path.basename(baseDir)
            data.file_pattern = "*"+feature+"*.npy"
            if withMS:
                data.dimensions.extend([self.featureGroupsIndex[i+1]-self.featureGroupsIndex[i]])
            else:
                dimensions = 0
                for entry in self.featureGroupsDict[feature]:
                    dimensions = dimensions + entry[1] - entry[0]
                data.dimensions.extend([dimensions])
            dataPb.data.extend([data]) 
            
        data = deepnet_pb2.Dataset.Data()
        data.name = "label_" + os.path.basename(baseDir) 
        data.dimensions.extend([1]) 
        data.file_pattern = "*label.npy"
        dataPb.data.extend([data]) 
        
        if withMS:
            MS = "withMS"
            outputProtoFile = os.path.join(baseDir, MS, "data_withMS.pbtxt")
        else:
            MS = "withoutMS"
            outputProtoFile = os.path.join(baseDir, MS, "data_withoutMS.pbtxt")
            
        dataPb.name = os.path.basename(baseDir) + "_"+ MS       
        dirPath = os.path.join(baseDir, MS)
        dataPb.prefix = dirPath
        for fileEntry in files:
            tempData = np.load(fileEntry)
            if len(tempData.shape) == 1 or tempData.shape[1] != 17593:
                continue
            instanceCount = instanceCount + tempData.shape[0]
            baseName = os.path.basename(fileEntry)
            fileName = os.path.join(dirPath,os.path.splitext(baseName)[0]) + "_" + MS
            np.save(fileName + '_label.npy', tempData[:, 17592])
            if withMS:
                for i, feature in enumerate(self.featureGroups):
                    np.save(fileName + '_' + feature + "_withMS.npy", tempData[:, self.featureGroupsIndex[i]:self.featureGroupsIndex[i + 1]])               
            else:
                for feature in self.featureGroups:
                    tempTuple = self.featureGroupsDict[feature][0]
                    tempArray = tempData[:, tempTuple[0]: tempTuple[1]]
                    if len(self.featureGroupsDict[feature]) > 1:
                        for i in range(1, len(self.featureGroupsDict[feature])):
                            tempTuple = self.featureGroupsDict[feature][i]
                            tempArray = np.concatenate((tempArray, tempData[:,tempTuple[0]: tempTuple[1]]), axis = 1)
                    np.save(fileName + '_' + feature + "_withoutMS.npy", tempArray) 
        for entry in dataPb.data:
            entry.size = instanceCount
        with open(outputProtoFile, 'w') as f:
            text_format.PrintMessage(dataPb, f) 
def main():
    baseDir = sys.argv[1]
    withMS = False if sys.argv[2].upper() == "FALSE" else True
    person = sys.argv[3].upper()
    if len(sys.argv) > 4:
        ne = sys.argv[4].upper()
    parser = FeatureParser()
    if person == 'FALSE':
        parser.ParseOther(baseDir,withMS = withMS)
    else:
        if ne == 'FALSE':
            parser.ParsePerson(baseDir, False, withMS = withMS)
        else:
            parser.ParsePerson(baseDir, True, withMS = withMS)
      
if __name__ == '__main__':
    main()        

