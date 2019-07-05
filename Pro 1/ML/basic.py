# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:22:52 2019

@author: Tony
"""
import numpy as np
import operator

class basic:
    def __init__(self):
        pass
    # read text file
    def read_txt(*params):
        input_file = params[0]
        with open(input_file, 'r', newline= '') as filereader:
            for row in filereader:
                print('{}'.format(row.strip()))
                
    # change the img to vector
    def img2Vector(self, *params):
        input_file = params[0]
        vector = np.zeros((1, 1024))
        count = 0
        with open(input_file, 'r', newline='') as filereader:
            for row in filereader:
                line = row.strip();
                for i in range(len(line)):
                    vector[0, len(line)*count + i] = int(line[i])
        return vector
    
    # knn classification algorithm
    # return the classified label
    def KNNClassification(self, *params):
        """
        参数: 
        - testVector: 用于分类的输入向量
        - trainingVectors: 输入的训练样本集
        - labels: 样本数据的类标签向量
        - k: 用于选择最近邻居的数目
        """
        testVector = params[0]
        trainingVectors = params[1]
        lables = params[2]
        k = params[3]
        
        dataSetSize = trainingVectors.shape[0]
        
        # get the difference between the test dataset and training dataset
        diffMat = np.tile(testVector, (dataSetSize, 1)) - trainingVectors
        
        sqdiffMat = diffMat**2
        sqDistances = sqdiffMat.sum(axis = 1)
        
        Distances = sqDistances**0.5
        
        sortedDistanceIndices = Distances.argsort()
        classCount = {}
        
        for i in range(k):
            classLabel = lables[sortedDistanceIndices[i]]
            classCount[classLabel] = classCount.get(classLabel, 0) + 1
            
        sortedClassCount = sorted(classCount.items(), key=lambda asd:asd[0], reverse=True)
        
        return sortedClassCount[0][0]