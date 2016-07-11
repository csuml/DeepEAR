"""Collects results from multiple runs and puts them into a nice table."""
import sys
import numpy as np
from deepnet import util
import os

def main():
  path = sys.argv[1]
  output_file = sys.argv[2]
  #import pdb
  #pdb.set_trace()
  layers = ['ZCR', 'MFCC', 'Spectral', 'Energy',
            'Chroma', 'PLP']
  MS = ['withMS', 'withoutMS']
  predsMap = {}
  f = open(output_file, 'w')
  f.write('\\begin{tabular}{|c|c|c|c|c|c|c} \\hline \n')
  f.write('MS &ZCR &MFCC &Spectral &Energy &Chroma &PLP \\\\ \\hline\n')
  for ms in MS:
    for layer in layers:
      mfile = os.path.join(path, ms, '%s_dnn_BEST' % layer)
      if not os.path.exists(mfile):
        if ms not in predsMap:
          predsMap[ms] = []
        predsMap[ms].append(0.0)
        continue
      model = util.ReadModel(mfile)
      preds = model.test_stat_es.correct_preds/model.test_stat_es.count
      if ms not in predsMap:
        predsMap[ms] = []
      predsMap[ms].append(preds)

  for ms in MS:
	f.write(ms)
	for item in predsMap[ms]:
		f.write('&%.4f'% item)
	f.write('\\hline\n')
  f.write('\\end{tabular}\n')
  f.close()

if __name__ == '__main__':
  main()

