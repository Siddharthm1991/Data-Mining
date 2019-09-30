'''
FileName : DecisionTree.py
Description : Build a decision tree for a training dataset and different hyperparameter values.
The hyperparameter values are
* Number of levels
* Impurity threshold
* Impurity method
Author : Siddharth Mathiazhagan (mathiazhagan.1)

Command to execute program :  python DecisionTree.py -train_data train_data.txt -train_label train_label.txt -test_data test_data.txt -nlevels 1 -pthrd 0.2 -impurity gini
'''

# Import necessary modules
import argparse
import numpy as np
import math

class Node:
    def __init__(self):
        # List of indices to the data points
        self.data_idx = []
        # Impurity method used to calculated impurity
        self.impurity_method = ""
        # Feature on which the decision is implemented
        self.dfeature = -1
        # Impurity at a node
        self.impurity = -1
        # Level of the node
        self.nlevels = -1
        # Number of features
        self.nfeatures = -1
        # Majority class label
        self._class = -1
        # Left child
        self.left_child = None
        # Right Child
        self.right_child = None
        # Major class at the node
        self.majorClass = -1

    '''Method to initialize the node with its data indices, the impurity method to use, its level in the tree
    and the number of features in the dataset
    Return Value :  A newly created node'''
    def initNode(self, data_idx, impurity_method, level):
        root = Node()
        root.data_idx = data_idx
        root.impurity_method = impurity_method
        root.nlevels = level
        root.nfeatures = trainData.shape[1]
        return root

    '''Method to calculate to impurity of a node using GINI index
    Return Value : Impurity Value (Float)'''
    def calculateGINI(self, data_idx):
        tot = len(data_idx)
        if(tot == 0):
            return 0
        classLabels = list(range(1, 6))
        ginidata = trainLabel[data_idx]
        totalProb = 0
        for i in classLabels:
            classCount = len(ginidata[ginidata == i])
            totalProb += ((classCount / tot) ** 2)
        return 1 - totalProb

    '''Method to calculate to impurity of a node using entropy
    Return Value : Impurity Value (Float)'''
    def calculateEntropy(self, data_idx):
        tot = len(data_idx)
        if (tot == 0):
            return 0
        classLabels = list(range(1, 6))
        ginidata = trainLabel[data_idx]
        totalProb = 0
        for i in classLabels:
            classCount = len(ginidata[ginidata == i])
            prob = classCount / tot
            if(prob != 0):
                totalProb += (prob * math.log(prob , 2))
        return -totalProb

    '''Util method to corresponding method based on the impurity type
    Return Value : Impurity Value (Float)'''
    def calculateIP(self, data_idx):
        if self.impurity_method == 'gini':
            P = self.calculateGINI(data_idx)
        else:
            P = self.calculateEntropy(data_idx)
        return P

    '''Method to build the tree
    Return Value : Root of decision tree (Node)'''
    def buildDT(self, label, impurity_method, p, nl, data=None):
        data_idx = np.arange(len(label))
        DT = self.initNode(data_idx, impurity_method, 0)
        DT.impurity = DT.calculateIP(DT.data_idx)
        DT.splitNode(nl, p)
        return DT

    '''Get indices of train data that match the value passed
    Return Value : List of Indices
    '''
    def getIndices(self,data_idx, feature , value):
        result = []
        for index in data_idx:
            if trainData[index , feature] == value:
                result.append(index)
        return result

    '''Method to calculate impurity at a node for each feature and then choose the feature which provides the maximum
     gain. Then, recursively call the split node function on the left child for the indices which have value 0 and 
     on the right child for the indices which have value 1'''
    def splitNode(self, nl, p):
        if self.nlevels < nl and self.impurity > p:
            maxGain = -1
            splitFeature = -1
            data = trainData[self.data_idx]
            bestleft = 0
            bestright = 0
            Gains = []
            for i in range(self.nfeatures):
                featData = data[:, i]
                totCount = len(featData)
                leftDataIdx = self.getIndices(self.data_idx , i , 0)
                Pleft = self.calculateIP(leftDataIdx)
                rightDataIdx = self.getIndices(self.data_idx , i , 1)
                Pright = self.calculateIP(rightDataIdx)
                leftCount = len(leftDataIdx)
                rightCount = len(rightDataIdx)
                M = (Pleft * (leftCount / totCount)) + (Pright * (rightCount / totCount))
                Gain = self.impurity - M
                Gains.append(Gain)
                if Gain > maxGain:
                    maxGain = Gain
                    splitFeature = i
                    bestleft = Pleft
                    bestright = Pright
            self.dfeature = splitFeature
            data_idx_left = self.getIndices(self.data_idx , self.dfeature , 0)
            data_idx_right = self.getIndices(self.data_idx , self.dfeature , 1)

            if len(data_idx_left) > 0:
                self.left_child = self.initNode(data_idx_left, self.impurity_method, self.nlevels + 1)
                self.left_child.impurity = bestleft
                leftLabel = trainLabel[data_idx_left]
                self.left_child.majorClass = np.bincount(leftLabel).argmax()
                self.left_child.splitNode(nl, p)

            if len(data_idx_right) > 0:
                self.right_child = self.initNode(data_idx_right, self.impurity_method, self.nlevels + 1)
                self.right_child.impurity = bestright
                rightLabel = trainLabel[data_idx_right]
                self.right_child.majorClass = np.bincount(rightLabel).argmax()
                self.right_child.splitNode(nl, p)

    '''Method to classify the data from the test dataset and write the output to an output file'''
    def classify(self, test_data, output_file, root):
        f = open(output_file , 'w+')
        classifyLabel = []
        for row in test_data:
            temp = root
            while temp:
                val = row[temp.dfeature]
                if val == 0:
                    if(temp.left_child):
                        temp = temp.left_child
                    else:
                        break
                else:
                    if(temp.right_child):
                        temp = temp.right_child
                    else:
                        break
            classifyLabel.append(temp.majorClass)
            f.write(str(temp.majorClass))
            f.write('\n')
        return classifyLabel

'''Build the confusion matrix using Class 1 as positive and the other classes as negative. Calculate the precision, 
recall and accuracy'''
def calcPrecisionRecall(output_file, testLabel):
    confusionMat = np.zeros((2,2))
    outputData = np.genfromtxt(output_file).astype(int)
    for i, val in enumerate(outputData):
        if(val == 1):
            if(testLabel[i] == 1):
                confusionMat[0,0] += 1
            else:
                confusionMat[1,0] += 1
        else:
            if(testLabel[i] == 1):
                confusionMat[0,1] += 1
            else:
                confusionMat[1,1] += 1
    tp = confusionMat[0,0]
    fn = confusionMat[0,1]
    fp = confusionMat[1,0]
    tn = confusionMat[1,1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Precision : ",precision)
    print("Recall : ",recall)
    print("Accuracy : ",accuracy)

'''Declare the parser and define the name of the command line arguments to be given by the user
Return Value : Parser'''
def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data')
    parser.add_argument('-train_label')
    parser.add_argument('-test_data')
    parser.add_argument('-nlevels')
    parser.add_argument('-pthrd')
    parser.add_argument('-impurity')
    parser.add_argument('-pred_file')
    return parser


if __name__ == '__main__':
    # Read the arguments using the argument parser
    parser = getParser()
    args = parser.parse_args()
    # Name of train data file
    trainDataFileName = str(args.train_data)
    # Name of test data file
    testDataFileName = str(args.test_data)
    # Name of train label file
    trainLabelFileName = str(args.train_label)
    # Number of levels in the tree
    numLevels = int(args.nlevels)
    # Impurity threshold
    purityThreshold = float(args.pthrd)
    # Impurity Type (Gini or Threshold)
    impurityType = str(args.impurity)
    # Filename for the output file
    outputFileName = str(args.pred_file)

    # Read the train data into a matrix
    trainData = np.genfromtxt(trainDataFileName, delimiter=' ').astype(int)
    # Read the train label into a matrix
    trainLabel = np.genfromtxt(trainLabelFileName, delimiter=' ').astype(int)
    # Read the test data into a matrix
    testData = np.genfromtxt(testDataFileName, delimiter=' ').astype(int)
    # Read the test labels into a matrix (To test accuracy, precision and recall)
    testLabel = np.genfromtxt('test_label.txt', delimiter=' ').astype(int)

    # Calling the build DT method
    util = Node()
    root = util.buildDT(label=trainLabel, impurity_method=impurityType, p=purityThreshold, nl=numLevels)
    # Calculating Training Accuracy
    trainClassify = root.classify(trainData , outputFileName , root)
    trainClassify = np.array(trainClassify)
    trainCorrect = np.sum(trainClassify == trainLabel)
    print('Train Accr : ',trainCorrect / len(trainLabel))
    # Calculating Testing Accuracy
    classifyLabels = root.classify(testData, outputFileName, root)
    classifyLabels = np.array(classifyLabels)
    correctlyClassified = np.sum(classifyLabels == testLabel)
    accuracy = correctlyClassified / len(testLabel)
    print("Test Accuracy : ", accuracy)
    # Calculating precision and recall
    calcPrecisionRecall(outputFileName , testLabel)
