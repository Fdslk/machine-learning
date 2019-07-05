# -*- coding: utf-8 -*-
import basic
from os import listdir
import numpy as np

if __name__ == "__main__":
    bs = basic.basic()
    knownLables = []
    input_path = '../data/digits/trainingDigits'
    input_test_path = '../data/digits/testDigits'
    trainingFileList = listdir(input_path)
    m = len(trainingFileList)
    
    trainingMat = np.zeros((m, 1024))
    
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classLabel = int(fileStr.split('_')[0])
        knownLables.append(classLabel)
        
        trainingMat[i, :] = bs.img2Vector(input_path + '/%s' % fileNameStr)
        
    testFileList = listdir(input_test_path)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classLabel = int(fileStr.split('_')[0])
        
        testVector = bs.img2Vector(input_test_path + '/%s' % fileNameStr)
        
        result = bs.KNNClassification(testVector, trainingMat, knownLables, 3)
        
        print("测试样本 %s, 预测分类%d, 真实分类%d" % (fileNameStr, result, classLabel))
        
        if(result != classLabel):
            errorCount += 1
    print("\n分类错误率：%.2f" % (errorCount/float(mTest)))