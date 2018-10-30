# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 18:58:03 2017

@author: PC
"""
import ML as ml
# calculate the fpr and tpr for all thresholds of the classification
probs = ml.model.predict_proba(ml.X_validation)
preds = probs[:,1]
fpr, tpr, threshold = ml.roc_curve(ml.Y_validation, preds)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % ml.roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()