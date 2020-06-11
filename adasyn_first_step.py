import numpy as np
from sklearn import neighbors
import pandas as pd
from sklearn.model_selection import train_test_split
import operator
from statistics import mean


def adasyn_first_step(xtrain, ytrain, target_column, class_to_boost, nominal, n_neighbors, boost_coef):
    # (xtrain, ytrain, beta, threshold, target_column, boost_coef, K=5)
    # we introduce the parameter class weight
    # it says how many times we want to increase the population of each class
    # df, X_train, y_train, class_weight, "target"
    train_dataset = pd.concat([xtrain, ytrain], axis=1, sort=False)
    # print(len(train_dataset))
    # print(train_dataset)
    if class_to_boost == 1:
        train_dataset = train_dataset.sort_values(by=target_column, ascending=False)
        m = int(sum(ytrain))
        # print(m)

    else:
        train_dataset = train_dataset.sort_values(by=target_column, ascending=True)
        m1 = int(sum(ytrain))
        # print(m1)
        m = len(ytrain) - m1
        # print(m)

    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(xtrain, ytrain)

    # Step 2a, if the minority data set is below the maximum tolerated threshold, generate data.
    # Beta is the desired balance level parameter.  Beta > 1 means u want more of the imbalanced type, vice versa.
    G = boost_coef * m - m

    # Step 2b, find the K nearest neighbours of each minority class example in euclidean distance.
    # Find the ratio ri = majority_class in neighbourhood / K
    Ri = []
    Minority_per_xi = []
    for i in range(m):
        xi = xtrain.iloc[i, :]
        # print(xi)
        # Returns indices of the closest neighbours, and return it as a list
        neighbours = clf.kneighbors([xi], n_neighbors=n_neighbors, return_distance=False)[0]
        # print(neighbours)
        # Skip classifying itself as one of its own neighbours
        # neighbours = neighbours[1:]

        # Count how many belongs to the majority class
        count = 0
        for value in neighbours:
            if value > m:
                count += 1

        # Find all the minority examples
        minority = []
        for value in neighbours:
            # Shifted back 1 because indices start at 0
            if value <= m - 1:
                minority.append(value)
        # print(minority)
        # print(count)
        if len(minority) >= 2:
            Ri.append(count / n_neighbors)
            Minority_per_xi.append(minority)
        elif len(minority) == 1:
            Ri.append(1/n_neighbors)
            Minority_per_xi.append(minority)
        else:
            Ri.append(0)
            Minority_per_xi.append(minority)

    # Step 2c, normalize ri's so their sum equals to 1
    Rhat_i = []
    for ri in Ri:
        rhat_i = ri / sum(Ri)
        Rhat_i.append(rhat_i)

    # Step 2d, calculate the number of synthetic data examples that will be generated for each minority example
    Gi = []
    for rhat_i in Rhat_i:
        gi = round(rhat_i * G)
        Gi.append(int(gi))
    # print(max(Gi))

    l = []
    for group in Minority_per_xi:
        l.append(len(group))

    # print(min(l))
    # # Step 2e, generate synthetic examples
    number_of_added_data = 0
    syn_data = []
    most_common = {}
    for i in range(m):
        most_common_nominal = {}
        xi = xtrain.iloc[i, :]
        if len(nominal) >= 1:
            for feature in nominal:
                count = 0
                sum_nominal = 0
                # print(feature)
                for sample in Minority_per_xi[i]:
                    # print(sample)
                    x_sample = xtrain.iloc[sample, :]
                    # print(x_sample)
                    # print(type(x_sample))
                    #feature_value = x_sample[feature]
                    sum_nominal += x_sample[feature]

                    #if feature_value in count:
                    #    count[feature_value] += 1
                    #else:
                    #    count[feature_value] = 1
                most_common_nominal[feature] = round(sum_nominal/len(Minority_per_xi[i]))
                #key_max = max(count.items(), key=operator.itemgetter(1))[0]
                #most_common_nominal[feature] = key_max
        most_common[i] = most_common_nominal
        # print("xi", xi)


    return Minority_per_xi, Gi, most_common, m
