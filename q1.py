import sys
from statistics import mode
import math
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)

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
    def GetNeighbores(Xtrain, x, K, ignorSelf=False):
        dist = []
        friends = []

        for i in range(Xtrain.shape[0]):
            if (ignorSelf and (Xtrain[i] == x).all()):
                ignoreSelf = False
                continue

            ABS = KNN.ABS_dist(Xtrain[i],x)
            #ed = KNN.Euclid_dist(Xtrain[i], x)
            dist.append((i, ABS))

        dist.sort(key=lambda item: item[1])
        for k in range(K):
            friends.append(dist[k][0])

        return friends

    def LabelClass(neighbors, Ytrain):
        labelVote = []
        for i in neighbors:
            labelVote.append(Ytrain[i])

        quorum = mode(labelVote)
        #quorum = max(labelVote, key=labelVote.count)
        return quorum

    def CalculateKNN(self, K, ignoreSelf=False):
        tr_senate = []
        valtr_senate = []
        te_senate = []

        for i in range(self.Xtrain.shape[0]):
            neighbores = KNN.GetNeighbores(self.Xtrain, self.Xtrain[i], K)
            vote = KNN.LabelClass(neighbores, self.Ytrain)
            tr_senate.append(vote)

        for i in range(self.Xtrain.shape[0]):
            neighbores = KNN.GetNeighbores(self.Xtrain, self.Xtrain[i], K, True)
            vote = KNN.LabelClass(neighbores, self.Ytrain)
            valtr_senate.append(vote)

        for i in range(self.Xtest.shape[0]):
            neighbores = KNN.GetNeighbores(self.Xtest, self.Xtest[i], K)
            vote = KNN.LabelClass(neighbores, self.Ytest)
            te_senate.append(vote)

        return tr_senate, valtr_senate, te_senate

    def CalculateError(self, Values, Y):
        delta = Values - Y
        err = len([val for val in delta if not math.isclose(val, 0.0)])
        return (err / len(Y)) * 100

def main():
    FILES = LoadData(sys.argv[1], sys.argv[2])

    Xtrain, Ytrain = FILES.GetTrain()
    Xtest, Ytest = FILES.GetTest()

    knn = KNN(Xtrain, Ytrain, Xtest, Ytest)
    tr_rep, valtr_rep, te_rep = knn.CalculateKNN(int(sys.argv[3]))

    tr_e = knn.CalculateError(tr_rep, Ytrain)
    valtr_e = knn.CalculateError(valtr_rep, Ytrain)
    te_e = knn.CalculateError(te_rep, Ytest)

    print("Training Error: {0:.4f}".format(tr_e))
    print("Validation Error: {0:.4f}".format(valtr_e))
    print("Testing Error: {0:.4f}".format(te_e))

    graph = input("Would you like to graph these three errors? [y/n]: ")
    graph.lower()
    if (graph == 'y' or graph == 'yes'):
        for i in range(1, 51, 3):
            tr_rep, valtr_rep, te_rep = knn.CalculateKNN(int(sys.argv[3]))

            tr_e = knn.CalculateError(tr_rep, Ytrain)
            valtr_e = knn.CalculateError(valtr_rep, Ytrain)
            te_e = knn.CalculateError(te_rep, Ytest)

            plt.plot(i, tr_e, 'bo')
            plt.plot(i, valtr_e, 'go')
            plt.plot(i, te_e, 'ro')

        try:
            plt.show()
        except:
            print("ERROR: Sorry cannot show the graph on your system.")

        plt.savefig("KNN_graph.png")
main()


