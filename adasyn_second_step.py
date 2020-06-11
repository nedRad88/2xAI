import numpy as np
from sklearn import neighbors
import pandas as pd
from sklearn.model_selection import train_test_split
import operator
from statistics import mean


def adasyn_second_step(xtrain, ytrain, target_column, m, most_common, complex_model, nominal, Gi, Minority_per_xi):
    # (xtrain, ytrain, beta, threshold, target_column, boost_coef, K=5)
    # we introduce the parameter class weight
    # it says how many times we want to increase the population of each class
    # df, X_train, y_train, class_weight, "target"
    train_dataset = pd.concat([xtrain, ytrain], axis=1, sort=False)
    # print(len(train_dataset))
    # print(train_dataset)
    syn_data = []
    for i in range(m):
        xi = xtrain.iloc[i, :]
        most_common_feature = most_common[i]
        # print("xi", xi)
        for j in range(Gi[i]):
            # If the minority list is not empty
            if Minority_per_xi[i]:
                index = np.random.choice(Minority_per_xi[i])
                xzi = xtrain.iloc[index, :]
                si = xi + (xzi - xi) * np.random.uniform(0, 1)
                if len(nominal) >= 1:
                    for feature in nominal:
                        si[feature] = most_common_feature[feature]
                syn_data.append(si)

    # Build the data matrix
    new_y = []
    for i in range(len(syn_data)):
        new_y.append(int(complex_model.predict([syn_data[i]])))

    new_y_df = pd.DataFrame({target_column: new_y})
    new_df = pd.DataFrame(syn_data)
    new_df.reset_index(drop=True, inplace=True)
    new_df = pd.concat([new_df, new_y_df], axis=1)

    return new_df
