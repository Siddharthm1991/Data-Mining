# Import necessary modules
import argparse
import itertools
import copy
import numpy as np
import math
import pandas as pd
import time

def readFile(fName):
    transactionData = {}
    i = 0
    for l in open(fName, 'r'):
        # print(l)
        if(i > 0):
            # print("IN")
            l = l.strip()
            data = l.split(" ")
            data = [int(x) for x in data]
            transactionData[i] = data
        # print("I = ",str(i))
        i += 1
    # print(transactionData)
    return transactionData

def generate_F1(database , minsupp):
    l1items = {}
    resSet = set()
    n = len(database)
    for k , v in database.items():
        for val in v:
            try:
                l1items[val] += 1
            except:
                l1items[val] = 1

    for k , v in l1items.items():
        sup = v
        # print(k)
        # print(sup)
        if(sup >= minsupp):
            val = frozenset([k])
            resSet.add(val)
    return resSet

def generateCandiadateKMinus1(fkItemSet,k):
    candSet = set()
    for entry in fkItemSet:
        # a = set(entry)
        for val in fkItemSet:
            # b = set(val)
            c = entry.union(val)
            if(len(c) == k):
                candSet.add(frozenset(c))

    return candSet

def generateCandidate(fkItemSet , f1ItemSet, k):
    candSet = set()
    for entry in fkItemSet:
        for val in f1ItemSet:
            if(isinstance(entry ,int)):
                entryList = [entry]
                entryList = set(entryList)
            else:
                entryList = set(entry)
            # if val not in entryList:
            val = set(val)
            for item in val:
                entryList.add(item)
            # entrySet = set(entryList)
            if(len(entryList) == k):
                candSet.add(frozenset(entryList))
    # candSet.sort()
    # res = list(val for val, _ in itertools.groupby(candSet))
    return candSet

def pruneCandidate(lk1ItemSet, fkItemSet, k):
    # print(fkItemSet)
    resCand = []
    for entry in lk1ItemSet:
        resSet = frozenset(itertools.combinations(entry , k - 1))
        prune = True
        for val in resSet:
            setVal = frozenset(val)
            # print(setVal)
            if setVal in fkItemSet:
                # print("IN")
                prune = False

        if prune != True:
            resCand.append(entry)

    return resCand

def countSupport(database, lk1ItemSet):
    resDict = {}
    n = len(database)
    # print("Candidate Items Length : "+str(len(lk1ItemSet)))
    for entry in lk1ItemSet:
        count = 0
        for k , v in database.items():
            itrCount = 0
            # dEntry = set(v)
            # cEntry = set(entry)
            if entry.issubset(v):
                count += 1
            itrCount += 1
            # if itrCount % 500 == 0:
                # print(str(itrCount) + " transactions processed")
        resDict[entry] = count
    return resDict

def outputFreqItemsets(supportCount , minsupp):
    resSet = set()
    for k , v in supportCount.items():
        if v >= minsupp:
            resSet.add(k)

    resSet = frozenset(resSet)
    return resSet

def apriori(database, minsupp, output_file):
    k = 2
    f1ItemSet = generate_F1(database, minsupp)
    fkItemSet = f1ItemSet
    while len(fkItemSet) > 0:
        startGenerate = time.time()
        lk1ItemSet = generateCandidate(fkItemSet , f1ItemSet, k)
        # lk1ItemSet = generateCandiadateKMinus1(fkItemSet, k)
        endGenerate = time.time() - startGenerate
        # print("Time to generate candidates : " + str(endGenerate))
        # print("Generated "+str(k)+"-itemsets")
        # if k == 3:
        #     print("L K + 1")
        #     print(lk1ItemSet)
        startPrune = time.time()
        prunedCand = pruneCandidate(lk1ItemSet, fkItemSet, k)
        endPrune = time.time() - startPrune
        # print("Time to prune candidates : " + str(endPrune))
        # print("Pruned " + str(k) + "-itemsets")
        # if k == 3:
        #     print("Pruned")
        #     print(prunedCand)
        startSupport = time.time()
        supportCount = countSupport(database, prunedCand)
        endSupport = time.time() - startSupport
        # print("Time to find support counts : "+str(endSupport))
        # if k == 3:
        # print(supportCount)
        # print("Calculated support of "+ str(k) + "-itemsets")
        fkItemSet = outputFreqItemsets(supportCount, minsupp)
        if(len(fkItemSet) > 0):
            resItemSet = fkItemSet
        # print("Frequent "+str(k)+"-itemset")
        # print("----------------------------")
        # print(fkItemSet)
        # fkItemSet = prunedCand
        k += 1
    # print("Frequent "+str(k - 2)+"-itemset")
    # print("----------------------------")
    # print(resItemSet)
    print(len(resItemSet))


'''Declare the parser and define the name of the command line arguments to be given by the user
Return Value : Parser'''
def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-database_file')
    parser.add_argument('-minsupp')
    parser.add_argument('-output_file')
    return parser

if __name__ == '__main__':
    # Read the arguments using the argument parser
    startTime = time.time()
    parser = getParser()
    args = parser.parse_args()
    dbName = str(args.database_file)
    minSupp = float(args.minsupp)
    outputFile = str(args.output_file)
    database = readFile(dbName)
    minSupp = minSupp * len(database)
    # print('Support : '+str(minSupp))
    import timeit
    print(timeit.timeit(lambda: apriori(database, minSupp, outputFile), number=1))
    # finish = time.time() - startTime
    # print(str(finish))

