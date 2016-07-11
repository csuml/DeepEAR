'''
Created on Mar 11, 2016

@author: ningzhang
'''
"""Do the online test for with model trained already"""
import sys
from neuralnet import *
from trainer import *
from neuralnet import *
from fastdropoutnet import *
from dbm import *
from dbn import *
from sparse_coder import *
from choose_matrix_library import *
import numpy as np
from time import sleep

def PrepareModel(model_file, train_op_file, eval_op_file):
    
  model, train_op, eval_op = LoadExperiment(model_file, train_op_file,
                                            eval_op_file)
  model = CreateDeepnet(model, train_op, eval_op)
  filelist = sys.argv[4].split()
  sampleList =[ np.load(fileName) for fileName in filelist ]
  batches = sampleList[0].shape[0]/model.batchsize
  model.t_op.randomize = False
  model.t_op.get_last_piece = True
  model.LoadModelOnGPU()
  return model

def OnlineTest(net, datalist, probability = False):
  """
  Net is the model object
  Datalist is required to be a list of numpy ndarray with order of net.inputlayer.
  Each ndarray in Datalist should be transposed version of design matrix: each 
  column of the matrix is one instance.
  Probability indicate if output the probability or not: 
  False, output classification label; True, output probability (raw output state)
  
  """
  if net.net.model_type == deepnet_pb2.Model.FEED_FORWARD_NET or \
        net.net.model_type == deepnet_pb2.Model.DBM:
      if len(net.input_datalayer) != len(datalist):
          raise Exception("Input data modality missing")
      for i, node in enumerate(net.input_datalayer):
          if node.data.shape != datalist[i].shape:
              raise Exception("Dimensionality error, matching layer %s failed" % (node.name))
          node.data.overwrite(datalist[i])
  if net.net.model_type == deepnet_pb2.Model.DBN:
      if len(net.upward_net.input_datalayer) != len(datalist):
          raise Exception("Input data modality missing")
      for i, node in enumerate(net.upward_net.input_datalayer):
          if node.data.shape != datalist[i].shape:
              raise Exception("Dimensionality error, matching layer %s failed" % (node.name))
          node.data.overwrite(datalist[i])                
  
  net.ForwardPropagate()
  
  res = []
  if net.net.model_type == deepnet_pb2.Model.FEED_FORWARD_NET or \
        net.net.model_type == deepnet_pb2.Model.DBM:
      for i, node in enumerate(net.output_datalayer):
          res.append(node.state.asarray())
          if not probability:
              res[i] = res[i] >= 0.5
  if net.net.model_type == deepnet_pb2.Model.DBN:
      for i, node in enumerate(net.rbm.output_datalayer):
          res.append(node.state.asarray())
          if not probability:
              res[i] = res[i] >= 0.5 

def main():
  board = LockGPU()
  model = PrepareModel(sys.argv[1], sys.argv[2], sys.argv[3])
  # filelist = sys.argv[4].split()
  data = sys.argv[4].values()
  # sampleList =[ np.load(fileName) for fileName in filelist ]
  # batches = sampleList[0].shape[0]/model.batchsize
  import pdb
  pdb.set_trace()
  # for i in range(batches):
  # batchsamples = [sample[i*model.batchsize: (i+1)*model.batchsize, :].transpose()] 
  res = OnlineTest(model, data)
  print res
  FreeGPU(board)


if __name__ == '__main__':
  main()

def test():
  print 'test'

