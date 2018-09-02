#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 03:59:00 2018

@author: nihagajam
"""
import sys
import sklearn
from sklearn.externals import joblib



def main():
    #f = open(sys.argv[2])
    f = open('foo.txt','r')
    fp = open('output.txt', 'w')

    vect = 'vector.sav'
    vectorizer = joblib.load(vect)

    model = 'logRegModel.sav'
    logisticRegr = joblib.load(model)


    string_ip = f.read()
        
    review_vector = vectorizer.transform([string_ip])
    result = logisticRegr.predict(review_vector)[0]
    
    fp.write(str(result))
    
    f.close()
    fp.close()


if __name__=='__main__':
    main()









