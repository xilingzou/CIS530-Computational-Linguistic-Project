import pandas as pd
from sklearn.metrics import f1_score

def getF1(filename):
  result = pd.read_csv(filename)
  result.columns = result.columns.str.lower()
  y_true = result['true']
  y_pred = result['pred']
  #calculate f1
  seperate_f1 = f1_score(y_true, y_pred, average =None)
  overall_f1 = f1_score(y_true, y_pred, average = 'macro')
  f1 = list(seperate_f1)
  f1.append(overall_f1)
  return f1
  
  
