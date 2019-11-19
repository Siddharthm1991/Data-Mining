# Import the required modules
import numpy as np
import argparse
import random

'''Read the data from the given database file.
Input Parameters: 
dbName -> Database file name
Returns: List of list (2D list)
'''
def read_data(dbName):
    db_data = []
    with open(dbName, 'r') as f:
        # Read the file line by line
        for line in f:
            line_list = line.split()
            # Convert the data points to type float
            num_list = [float(x) for x in line_list]
            db_data.append(num_list)

    db_data = np.array(db_data)
    return db_data

'''Calculate euciledean distance between the points given to the method
Input parameters : 
p1 -> Point 1
p2 -> Point 2 
Returns : Distance between the points, an float value
'''
def find_dist(p1 , p2):
    d = np.sqrt(np.sum(np.square(p1 - p2)))
    return d

'''Assign a cluster ID to every point in the data base by calculating its distance to the centroids and assigning the 
cluster ID based on which centroid is closes to the point
Input parameters: 
db_data -> 2D list of points read from the text database file
old_cent -> centroids of the k clusters
Returns:
cluster_data -> Dictionary where the key is the cluster ID and the value is the lists of points belonging to the cluster
cluster_idx_data -> Dictionary where the key is the cluster ID and the value is the lists of indexes for points 
belonging to the cluster
'''
def assign_clusters(db_data , old_cent):
    cluster_data = {}
    cluster_idx_data = {}
    for j , point in enumerate(db_data):
        minDist = float("inf")
        clusterNum = -1
        for i , centre in enumerate(old_cent):
            dist = find_dist(point , centre)
            # Check if the distance for point to a centroid is the smallest
            if(dist < minDist):
                clusterNum = i
                minDist = dist
        try:
            '''Maintaining a list of points to use for calculations in the subsequent iterations and
            maintaining a list of indices for writing to the output file'''
            cluster_data[clusterNum].append(point)
            cluster_idx_data[clusterNum].append(j)
        except:
            cluster_data[clusterNum] = [point]
            cluster_idx_data[clusterNum] = [j]
    return cluster_data , cluster_idx_data

'''Calculate the new centroid points by calculating the mean of the points that belong to a particular cluster
Input Parameters: 
cluster_data -> dictionary where the key is the cluster id and value is list of points corresponding to the cluster
Returns : List of new centroids'''
def calc_new_cent(cluster_data):
    res = []
    for k , v in cluster_data.items():
        points = np.array(v)
        # Calculate the mean of the points of the cluster across the columns
        meanVal = np.mean(points , axis=0)
        res.append(meanVal)
    res = np.array(res)
    return res

'''Calculate the maximum change between centroids of a cluster
Input Parameters:
new_cent -> list of new centroids for the clusters
old_cent -> list of old centroids for the clusters
Returns : an integer value which is the maximum distance between centroids of a cluster'''
def calc_cent_change_dist(new_cent , old_cent):
    max_dist = float("-inf")
    for i in range(new_cent.shape[0]):
        cent_dist = find_dist(new_cent[i] , old_cent[i])
        max_dist = max(max_dist , cent_dist)
    return max_dist

'''Write the data about which points belong to which cluster into a output file provided by the user
Input Parameters : 
outputFile -> Name of the output file provided by the user
cluster_data -> Data to be written to the file'''
def output_to_file(outputFile , cluster_data):

    with open(outputFile, 'w') as f:
        # Iterate through the dictionary , k -> cluster id , v -> list of points that belong to the cluster
        for k , v in cluster_data.items():
            cluster_points = " ".join(str(x) for x in v)
            f.write(str(k) + " : " + cluster_points)
            f.write('\n')

'''Function to generate the K clusters given the input data and the parameters
Input Parameters:
db_data -> List of points
k -> Number of clusters to be formed
max_iters -> Maximum number of iterations to be performed
eps -> Epsilon value that determines the minimum value of change between centroids for the iteration to continue
outputFile -> Name of the output file to which the result is written into
'''
def genKmeans(db_data , k , max_iters , eps , outputFile):
    # Give a seed to the random so that same data is the output of the random sampling
    random.seed(100)
    # Choose k indices randomly from the total number of points available
    rand_cent = random.sample(range(1 , len(db_data)) , k)
    old_cent = db_data[rand_cent]
    max_dist = float("inf")
    iter_count = 0
    cluster_idx_data = {}
    '''Keep running while loop as long the iteration count is less than the max iterations or the change between the 
    centroids is greater than epsilon'''
    while(max_dist > eps and iter_count < max_iters):
        # Call assign clusters to assign a cluster to each point
        cluster_data , cluster_idx_data = assign_clusters(db_data , old_cent)
        # Calculate the new centroids using the clustered data
        new_cent = calc_new_cent(cluster_data)
        # Find the maximum change distance between the new and old centroids
        find_cent_change_dist = calc_cent_change_dist(new_cent , old_cent)
        max_dist = find_cent_change_dist
        old_cent = new_cent
        iter_count += 1
    # Write the output to the output file
    output_to_file(outputFile , cluster_idx_data)


'''Declare the parser and define the name of the command line arguments to be given by the user
Return Value : Parser'''
def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-database_file')
    parser.add_argument('-k')
    parser.add_argument('-max_iters')
    parser.add_argument('-eps')
    parser.add_argument('-output_file')
    return parser

if __name__ == '__main__':
    # Read the arguments using the argument parser
    parser = getParser()
    args = parser.parse_args()
    # Name of the file to read transactions from
    dbName = str(args.database_file)
    # Value of K
    k = int(args.k)
    # Max Iteration Count
    max_iters = int(args.max_iters)
    # Epsilon
    eps = float(args.eps)
    # Name of output file to write results to
    outputFile = str(args.output_file)
    # Read the data from the database file
    db_data = read_data(dbName)
    # Generate the K clusters using the given parameters
    genKmeans(db_data , k , max_iters , eps , outputFile)
