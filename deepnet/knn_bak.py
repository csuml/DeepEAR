'''
Created on Apr 19, 2015

@author: ningzhang
'''
from neuralnet import *
from fastdropoutnet import *
from dbm import *
from dbn import *
from sparse_coder import *
from choose_matrix_library import *
from deepnet import trainer as tr
import numpy as np
import scipy.io as sio
from time import sleep

def Knn(patterns, targets, batchsize,K, labels, stats):
   if K > batchsize:
       batchsize = K
   n = targets.shape[0]
   num_batches = n/batchsize
   if targets.shape[0] - batchsize*num_batches >= K:
       num_batches = num_batches + 1
       
   pos = 0
   m = patterns.shape[0]
   dist = np.zeros((m,batchsize))
   dist_temp = cm.CUDAMatrix(np.zeros((m,batchsize)))
   pattern_mat = cm.CUDAMatrix(patterns)
   minDist_indices = np.zeros((m,K))
   neibor_labels = np.empty((m,K,labels.shape[1]))
   mean_mat = cm.CUDAMatrix(-stats['mean'].reshape(1,stats['mean'].shape[0]))
   std_mat = cm.CUDAMatrix(stats['std'].reshape(1,stats['std'].shape[0]))
   
   temp_mat = cm.CUDAMatrix(patterns)
   temp_mat_squre = cm.CUDAMatrix(patterns)
   temp_mat_squre.mult(temp_mat)
   sum_pattern = cm.sum(temp_mat_squre,1)
   dist_pattern_array = np.zeros((m,batchsize))
   dist_pattern_array[:,0] = sum_pattern.asarray().flatten()

   for j in range(batchsize-1):
       dist_pattern_array[:,j+1] = np.copy(dist_pattern_array[:,0])
   dist_pattern = cm.CUDAMatrix(dist_pattern_array)   
   #epsulo_mat = cm.CUDAMatrix(np.multiply(stats['std'],stats['std']))

   
   for i in range(num_batches):
       end = pos + batchsize
       if pos+batchsize > n:
           dist_temp = cm.CUDAMatrix(np.zeros((m,n-pos)))
           target_mat = cm.CUDAMatrix(targets[pos:end,:])
           dist_pattern = cm.CUDAMatrix(dist_pattern_array[:,0:n-pos])  
           end = n
       target_mat = cm.CUDAMatrix(targets[pos:end,:])
       # regularization
       target_mat.add_row_vec(mean_mat)
       target_mat.div_by_row(std_mat)
       dim = target_mat.shape[0]
       cm.dot(pattern_mat,target_mat.T,-2,dist_temp)
       
       target_mat.mult(target_mat)       
       sum_target = cm.sum(target_mat,1)
       dist_target_array = np.zeros((m,dim))
       dist_target_array[0,:] = np.transpose(sum_target.asarray()).flatten()

       for j in range(m-1):
           dist_target_array[j+1,:] = np.copy(dist_target_array[0,:])
                  
       dist_target = cm.CUDAMatrix(dist_target_array)  
       
       dist_temp.add(dist_pattern).add(dist_target)
        
       if i == 0 :
           temp_array = dist_temp.asarray()
           minDist_indices = temp_array.argsort()[:,0:K]
           temp_array.sort()
           dist = temp_array[:,0:K]
           for ind in range(m):
                neibor_labels[ind] = labels[minDist_indices[ind,:],:]
       else :
           temp_array = dist_temp.asarray()
           indice = temp_array.argsort() 
           temp_array.sort()
           dist_array = temp_array[:, 0:K]
           
           for ind_1 in range(m):
               for ind_2 in range(K):
                   for ind_3 in range(K):
                       if dist[ind_1, ind_3] > dist_array[ind_1,ind_2]:
                           dist[ind_1, ind_3] = dist_array[ind_1,ind_2]
                           minDist_indices[ind_1, ind_3] = indice[ind_1,ind_2] + pos
                           neibor_labels[ind_1, ind_3] = labels[minDist_indices[ind_1, ind_3],:]
                           break
                       
       pos = pos + batchsize
       target_mat.free_device_memory()
       dist_target.free_device_memory()
       
   temp_mat.free_device_memory()
   dist_pattern.free_device_memory()    
   temp_mat_squre.free_device_memory()     
       
   dist_temp.free_device_memory()
   pattern_mat.free_device_memory()
   mean_mat.free_device_memory()
   std_mat.free_device_memory()                   
   return dist, minDist_indices, neibor_labels        
           
               

def main():
  patternfile = sys.argv[1]
  targetfile = sys.argv[2]
  label_file = sys.argv[3]
  output_dir = sys.argv[4]
  statiticsFile = '/data1/ningzhang/flickr/flickr_stats.npz'
  
  batchsize = 128
  K = 5
  if len(sys.argv) > 5:
    K = sys.argv[5]
  if len(sys.argv) > 6:
    batchsize = sys.argv[6]
  else:
    gpu_mem = '2G'
  if len(sys.argv) > 6:
    main_mem = sys.argv[6]
  else:
    main_mem = '30G'
  import pdb
  pdb.set_trace()  
  board = tr.LockGPU()
  targets = np.load(targetfile)
  patterns = np.load(patternfile)
  labels = np.load(label_file)
  stats= np.load(statiticsFile)
  dist, minDist_indices, neibor_labels = Knn(patterns, targets, batchsize,K, labels,stats)
  
  dist_dir = os.path.join(output_dir,'distance')
  indices_dir = os.path.join(output_dir,'indices')
  labels_dir = os.path.join(output_dir,'labels')
  np.save(dist_dir, dist)
  np.save(indices_dir,minDist_indices)
  np.save(labels_dir, neibor_labels)
  sio.savemat(os.path.join(output_dir,'distance_mat'),{'distance':dist})
  sio.savemat(os.path.join(output_dir,'indices_mat'),{'indices':minDist_indices})
  sio.savemat(os.path.join(output_dir,'labels_mat'),{'labels':neibor_labels})
  tr.FreeGPU(board)
  
if __name__ == '__main__':
    main()
