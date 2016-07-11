"""Collects results from multiple runs and puts them into a nice table."""
import sys
import numpy as np
from deepnet import util
import os
import glob 
import pdb

def main():
  filePath = sys.argv[1]
  outputPath = sys.argv[2]
  classifiers = sorted(glob.glob(os.path.join(filePath,"*BEST")))
  f = open(outputPath, 'w')
  pdb.set_trace()
  for classifier in classifiers:
      baseName = os.path.basename(classifier)
      if "op" in baseName:
          continue
      model = util.ReadModel(classifier)
      testPreds = model.test_stat_es.correct_preds/model.test_stat_es.count
      trainPreds = model.train_stat_es.correct_preds/model.train_stat_es.count
      validPreds = model.best_valid_stat.correct_preds/model.best_valid_stat.count
      f.write('%s train: %.5f valid: %.5f test: %.5f \n' % 
              (os.path.basename(classifier),testPreds,trainPreds,validPreds))
                                                         
  f.close()

if __name__ == '__main__':
  main()

