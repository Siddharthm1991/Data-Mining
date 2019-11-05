# Import necessary modules
import argparse
import itertools
from collections import Counter

'''Method to read the input transaction file. The transactions are stored as list of sets so that operations 
to check if frequent item subsets are fast
Input Paramaters : fName -> File name for transaction data
Returns : A list of sets containing the transaction information'''
def readFile(fName):
    transactionData = []
    i = 0
    for l in open(fName, 'r'):
        if(i == 0):
            num_lines , num_items = l.split(" ")
        if(i > 0):
            l = l.strip()
            data = l.split(" ")
            # Convert the transaction numbers to an integer data
            data = [int(x) for x in data]
            transactionData.append(set(data))
        i += 1
    return num_lines , num_items , transactionData

'''Method to generate the 1-itemsets that have support count greater than the support threshold
Input Parameters : 
database -> List of sets of transactions
minsupp -> Minimum support threshold value
Returns: Set of frozensets of frequent 1-itemsets'''
def generate_F1(database , minsupp):

    result_list = [item for v in database for item in v]
    f1_counts = Counter(result_list)
    resSet = set([frozenset([k]) for k , v in f1_counts.items() if v >= minsupp])
    return resSet

'''Method to generate the L(K+1) candidate set given the FK frequent itemset. This is done by adding an element from
the F1 itemset to every itemset in FKitemset
Input Parameters : 
fkItemSet -> Frequent Itemsets of length k - 1
f1ItemSet -> Frequesnt Itemset of Length 1
k -> Length of itemset to be generated
Returns: Set of frozensets of itemsets of length k'''
def generateCandidate(fkItemSet , f1ItemSet, k):
    candSet = set()
    for entry in fkItemSet:
        for val in f1ItemSet:
            entryList = set(entry)
            # Add a value from the F1 itemset for the FK itemset
            for item in set(val):
                entryList.add(item)
            # Add itemset to result set only if the length of the itemset is K
            if(len(entryList) == k):
                candSet.add(frozenset(entryList))
    return candSet

'''Method to prune the candidates generated by the generateCandidate method. In this method we check if every k-1 length
subset is frequent. If not, we discard the itemset.
Input Paramaters : 
lk1ItemSet -> Frequent Itemsets of length k
fkItemSet -> Frequent Itemsets of length k - 1
k -> Length of itemset to be generated
Returns : Set of frozensets of itemsets of length k'''
def pruneCandidate(lk1ItemSet, fkItemSet, k):
    resCand = set()
    for entry in lk1ItemSet:
        # Method to get all subsets of length k -1 for a entry in L(K+1)
        resSet = frozenset(itertools.combinations(entry , k - 1))
        prune = True
        for val in resSet:
            setVal = frozenset(val)
            # Check if the subset is present in the k - 1 length frequent itemsets
            if setVal in fkItemSet:
                prune = False

        # If the itemset shouldn't be pruned add it to a result set that should be returned
        if prune != True:
            resCand.add(entry)

    return resCand

'''Method to count the support of the itemsets of length K
Input Parameters:
database -> List of sets of transactions
lk1ItemSet -> Frequent Itemsets of length k
Returns: A dictionary with the itemset as the key and its corresponding count as the value'''
def countSupport(database, lk1ItemSet):
    resDict = {}
    for entry in lk1ItemSet:
        count = 0
        for v in database:
            # Check if the itemset is a subset of a transaction
            if entry.issubset(v):
                count += 1
        resDict[entry] = count
    return resDict

'''Method to find all the k length itemsets which have a support count greater than the minimum support threshold
Input Parameters:
supportCount -> The dictionary returned by the countSupport method
minsupp -> Minimum support threshold
Returns: Set of fronzensets of the frequent itemsets of length k'''
def eliminateCandidates(supportCount , minsupp):
    resSet = set()
    for k , v in supportCount.items():
        if v >= minsupp:
            resSet.add(k)
    resSet = set(resSet)
    return resSet

'''Method to write all the frequent itemsets (F1, F2, .... Fk) to the output file
Input Parameters:
resItemSet -> List of all frequent itemsets
output_file -> Name of the output file
num_items -> Number of transaction items'''
def output_freq_itemsets(resItemSet, output_file, num_items):    
    with open(output_file , 'w') as f:
        f.write(str(len(resItemSet)) +" "+ str(num_items))        
        for value in resItemSet:
            resStr = " ".join(str(x) for x in value)
            f.write(resStr)
            f.write('\n')    
    print("Number of result itemsets : " + str(len(resItemSet)))  

'''Method to implement Apriori Algorithm 
Input Parameters:
database -> List of sets of transactions
minsupp -> Minimum support threshold
output_file -> Name of output file to write the results to'''
def apriori(database, minsupp, output_file, num_items):
    k = 2
    resItemSet = []
    # Generate the F1 itemsets
    f1ItemSet = generate_F1(database, minsupp)
    fkItemSet = f1ItemSet
    # Variable to store the result
    resItemSet.extend(f1ItemSet)
    # Run iterations as long as itemsets of length k is not zero
    while len(fkItemSet) > 0:
        # Generate candidate itemsets of length k
        lk1ItemSet = generateCandidate(fkItemSet , f1ItemSet, k)
        # Pruned candidates itemsets of length k
        prunedCand = pruneCandidate(lk1ItemSet, fkItemSet, k)
        # Get the support count for the itemsets of length k
        supportCount = countSupport(database, prunedCand)
        # Find the frequent items by selecting itemsets with count greater than the threshold
        fkItemSet = eliminateCandidates(supportCount, minsupp)
        if(len(fkItemSet) > 0):            
            resItemSet.extend(fkItemSet)            
        k += 1
    # Writing the result itemsets to a output file
    output_freq_itemsets(resItemSet, output_file, num_items) 


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
    parser = getParser()
    args = parser.parse_args()
    # Name of the file to read transactions from
    dbName = str(args.database_file)
    # Minimum support threshold value
    minSupp = float(args.minsupp)
    # Name of output file to write results to
    outputFile = str(args.output_file)
    num_lines , num_items , database = readFile(dbName)    
    minSupp = minSupp * len(database)
    # import timeit
    # Print time taken for program to execute
    # print(timeit.timeit(lambda: apriori(database, minSupp, outputFile, num_items), number=1))
    apriori(database, minSupp, outputFile, num_items)
