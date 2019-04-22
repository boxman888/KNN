import sys
from statistics import mode
import math
import numpy as np

class LoadData:
    def __init__(self, train, test):
        self.train = np.genfromtxt(train, delimiter=',')
        self.test = np.genfromtxt(test, delimiter=',')

    def Normalize(X):
        return (X-X.min(0)) / X.ptp(0);

    def GetTrain(self):
        data = LoadData.Normalize(self.train)
        x = data[:,1:]
        y = self.train[:,0]
        return x, y

    def GetTest(self):
        data = LoadData.Normalize(self.test)
        x = data[:,1:]
        y = self.test[:,0]
        return x, y

class KNN:
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest

    # Get the euclidean distance
    def Euclid_dist(v1, v2):
        return math.sqrt(np.sum(np.power((v1 - v2), 2)))

    def ABS_dist(v1, v2):
        return np.sum(np.absolute(v1 - v2))
    # Get Neighbors(Xtrain) near y, up to as K.
    def GetNeighbores(Xtrain, y, K):
        dist = []
        friends = []

        # Store location of Xtrain[i] with distance in a tuple
        # This will be used for looking up y's
        for i in range(Xtrain.shape[0]):
            ed = KNN.Euclid_dist(Xtrain[i], y)
            #ABS = KNN.ABS_dist(Xtrain[i],y)
            dist.append((i, ed))

        # Sort tuples based on the euclid distance from k.
        #dist.sort(key=KNN.SortTupleData)
        #print(dist)
        #print("before")
        dist.sort(key=lambda item: item[1])
        #print("after")
        #print(dist)
        #print("zero")
        #print(dist[0][0])
        for k in range(K):
            friends.append(dist[k][0])

        return friends

    # Utility funciton used for GetNeighbors function.
    #def SortTupleData(matrix):
    #    return matrix[:][1];

    def LabelClass(neighbors, Ytrain):
        labelVote = []
        for i in neighbors:
            labelVote.append(Ytrain[i])

        #print(neighbors)
        quorum = mode(labelVote)
        return quorum

    def CalculateKNN(self, K):
        tr_senate = []
        te_senate = []

        for i in range(self.Xtrain.shape[0]):
            neighbores = KNN.GetNeighbores(self.Xtrain, self.Ytrain[i], K)
            vote = KNN.LabelClass(neighbores, self.Ytrain)
            tr_senate.append(vote)

        for i in range(self.Xtest.shape[0]):
            neighbores = KNN.GetNeighbores(self.Xtest, self.Ytest[i], K)
            vote = KNN.LabelClass(neighbores, self.Ytest)
            te_senate.append(vote)
        #print(senate)
        return tr_senate, te_senate

    def CalculateError(self, Values, Y):
        delta = Values - Y
        err = len([val for val in delta if not math.isclose(val, 0.0)])
        return err / len(Y)

def main():
    FILES = LoadData(sys.argv[1], sys.argv[2])

    Xtrain, Ytrain = FILES.GetTrain()
    Xtest, Ytest = FILES.GetTrain()

    #print(Xtrain)
    knn = KNN(Xtrain, Ytrain, Xtest, Ytest)
    tr_rep, te_rep = knn.CalculateKNN(51)

    tr_e = knn.CalculateError(tr_rep, Ytrain)
    te_e = knn.CalculateError(te_rep, Ytest)

    print("Training Error: ", tr_e)
    print("Testing Error: ", te_e)

main()


