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
import glob

def Knn(patterns, targets, batchsize, K, labels = [], stats = []):
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
   if labels:
       neibor_labels = np.empty((m,K,labels.shape[1]))
   if stats:
       mean_mat = cm.CUDAMatrix(-stats['mean'].reshape(stats['mean'].shape[0],1))
       std_mat = cm.CUDAMatrix(stats['std'].reshape(stats['std'].shape[0],1))
       
   target_mat = cm.CUDAMatrix(np.zeros((targets.shape[1],batchsize)))
   sum_target = cm.CUDAMatrix(np.zeros((1,batchsize)))
   dim = batchsize
   temp_mat = cm.CUDAMatrix(patterns)
   temp_mat_squre = cm.CUDAMatrix(patterns)
   temp_mat_squre.mult(temp_mat)
   sum_pattern = cm.sum(temp_mat_squre,1)
   
   #epsulo_mat = cm.CUDAMatrix(np.multiply(stats['std'],stats['std']))

   
   for i in range(num_batches):
       end = pos + batchsize
       if pos+batchsize > n:
           end = n
           target_mat.free_device_memory()    
           dist_temp.free_device_memory()
           sum_target.free_device_memory()
           dist_temp = cm.CUDAMatrix(np.zeros((m,n-pos)))
           target_mat = cm.CUDAMatrix(targets[pos:end,:].T)
           sum_target = cm.CUDAMatrix(np.zeros(1,n-pos))
       else:    
           target_mat.overwrite(targets[pos:end,:].T)
       # regularization
       if stats:
           target_mat.add_col_vec(mean_mat)
           target_mat.div_by_col(std_mat)   

       cm.dot(pattern_mat,target_mat,-2,dist_temp)
       
       target_mat.mult(target_mat)
       cm.sum(target_mat,0,sum_target)
       """
       dist_pattern_array = np.zeros((m,dim))
       dist_target_array = np.zeros((m,dim))
       dist_pattern_array[:,0] = sum_pattern.asarray().flatten()
       dist_target_array[0,:] = np.transpose(sum_target.asarray()).flatten()

       for j in range(dim-1):
           dist_pattern_array[:,j+1] = dist_pattern_array[:,0]
       for j in range(m-1):
           dist_target_array[j+1,:] = dist_target_array[0,:]
           
       dist_pattern = cm.CUDAMatrix(dist_pattern_array)
       dist_target = cm.CUDAMatrix(dist_target_array)  
       
       dist_temp.add(dist_pattern).add(dist_target)
       """
       dist_temp.add_col_vec(sum_pattern).add_row_vec(sum_target)
       if i == 0 :
           temp_array = dist_temp.asarray()
           minDist_indices = temp_array.argsort()[:,0:K]
           temp_array.sort()
           dist = temp_array[:,0:K]
           if labels:
               for ind in range(m):
                    neibor_labels[ind] = labels[minDist_indices[ind,:],:]
       else :
           temp_array = dist_temp.asarray()
           indice = temp_array.argsort() 
           temp_array.sort()
           dist_array = temp_array[:, 0:K]
           K_new = K if K <= n-pos else n-pos
           dist_pool = np.zeros((1, K + K_new))
           for ind_1 in range(m):
               dist_pool[0, 0:K] = dist[ind_1, 0:K]
               dist_pool[0, K:K + K_new] = dist_array[ind_1, 0:K_new]
               internal_compare = dist_pool.argsort().flatten()
               dist[ind_1, :] = dist_pool[0,internal_compare[0:K]]
               for j in range(K):
                   minDist_indices[ind_1,j] = minDist_indices[ind_1,j] if internal_compare[j] < K else indice[ind_1,j] + pos
                   if labels:
                       neibor_labels[ind_1, j] = labels[minDist_indices[ind_1, j],:]
               """        
               for ind_2 in range(K):
                   for ind_3 in range(K):
                       if dist[ind_1, ind_3] > dist_array[ind_1,ind_2]:
                           dist[ind_1, ind_3] = dist_array[ind_1,ind_2]
                           minDist_indices[ind_1, ind_3] = indice[ind_1,ind_2] + pos
                           if labels:
                               neibor_labels[ind_1, ind_3] = labels[minDist_indices[ind_1, ind_3],:]
                           break
                """                
       pos = pos + batchsize
       sys.stdout.write('\rKNN: %d processed' % pos)
       sys.stdout.flush()
       
       #dist_pattern.free_device_memory()
       #dist_target.free_device_memory()
       
   temp_mat.free_device_memory()    
   temp_mat_squre.free_device_memory()     
   target_mat.free_device_memory()    
   dist_temp.free_device_memory()
   pattern_mat.free_device_memory()
   if stats:
       mean_mat.free_device_memory()
       std_mat.free_device_memory()
   if labels:                   
       return dist, minDist_indices, neibor_labels 
   else:
       return dist, minDist_indices       
           
               

def main():
  patternFilePattern = sys.argv[1]
  targetFilePattern = sys.argv[2]
  
  output_dir = sys.argv[3]
  if len(sys.argv) > 4:
      label_file = sys.argv[4]
  statiticsFile = '/data1/ningzhang/flickr/flickr_stats.npz'
  
  batchsize = 1000
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
  patternFiles = sorted(glob.glob(patternFilePattern))
  targetFiles = sorted(glob.glob(targetFilePattern))
  
  stats= np.load(statiticsFile)
  patternlist = []
  m = 0
  for i, patternFile in enumerate(patternFiles):
      patternlist.append(np.load(patternFile))
      m += patternlist[i].shape[0]
  patterns = np.zeros((m,patternlist[0].shape[1]))    
  pos = 0
  for patternShark in patternlist:
     patterns[pos: pos+patternShark.shape[0], :] = patternShark
     pos = pos + patternShark.shape[0]
  pos = 0
  dist_pool = np.zeros((1,2*K))
  
  if len(sys.argv) > 4:
      labels = np.load(label_file)
  for targetFile in targetFiles:
      targets = np.load(targetFile)
      if len(sys.argv) > 4:
          dist_interm, minDist_indices_interm, neibor_labels_interm = Knn(patterns, targets, batchsize,K, labels,stats)
      else:
          dist_interm, minDist_indices_interm = Knn(patterns, targets, batchsize, K)#, stats = stats)
          
      if pos == 0:
          dist = np.copy(dist_interm)
          minDist_indices = np.copy(minDist_indices_interm)
          if len(sys.argv) > 4:
              neibor_labels = np.copy(neibor_labels_interm)
      else:
          K_new = K if K <= targets.shape[0] else targets.shape[0]
          if K_new < K:
            dist_pool = np.zeros((1, K + K_new))
          for ind_1 in range(m):
               dist_pool[0, 0:K] = dist[ind_1, 0:K]
               dist_pool[0, K:K+K_new] = dist_interm[ind_1, 0:K_new]
               internal_compare = dist_pool.argsort().flatten()
               dist[ind_1, :] = dist_pool[0,internal_compare[0:K]]
               for j in range(K):
                   minDist_indices[ind_1,j] = minDist_indices[ind_1,j] if internal_compare[j] < K else minDist_indices_interm[ind_1,j] + pos
                   if len(sys.argv) > 4:
                       neibor_labels[ind_1, j] = labels[minDist_indices[ind_1, j],:] 
      
      pos = pos + targets.shape[0]
  
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
