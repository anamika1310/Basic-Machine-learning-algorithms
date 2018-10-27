# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 01:39:46 2018

@author: sweety
"""

#!/usr/bin/python

""" 
    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
t1=time()
clf.fit(features_train,labels_train)
print("training time:",round(time()-t1,3),"s")
t2=time()
l=clf.predict(features_test)
print("prediction time:",round(time()-t2,3),"s")
for i in l:
    if(i==0):
        print("author is: Sara",end=",")
    else:
        print("author is: Chris",end=",")
        
from sklearn.metrics import accuracy_score
print(accuracy_score(l,labels_test))  
print("10th:",l[10])
print("26:",l[26])
print("50th",l[50])              

#########################################################
### your code goes here ###


#########################################################


