from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from rich import print
import pdb
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# df_use = pd.read_csv('results/use_mixed.csv')
df_use = pd.read_csv('results/use_bn_mixed.csv')

y_test = np.array(df_use['GroundTruth'])
y_pred = np.array(df_use['Pred'])

print(
    classification_report(y_test,
                          y_pred,
                          target_names=['unknow', 'update', 'new']))
# new:	1295	327
# unknow:	426	109
# update:	3900	980

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat,
            annot=True,
            fmt='d',
            xticklabels=['unknow', 'update', 'new'],
            yticklabels=['unknow', 'update', 'new'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()