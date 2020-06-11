import graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from dtxai import *
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from math import sqrt, floor
import pickle


# output_file = open("results_2_step_cost_training_distance_heart_all4.txt", 'w')
files = ["sick"]
depths = [3]
class_weights = []
for i in range(5):
    class_weights.append(i + 1)
class_weight = len(class_weights)
trials = [i for i in range(20)]
plot_number = 1
complex_models = pickle.load(open("final_tests/NN/dtextract/sick/complex_models_nn_sick.p", "rb"), encoding='latin1')
training_sets = pickle.load(open("final_tests/NN/dtextract/sick/training_nn_sick.p", "rb"), encoding='latin1')
test_sets = pickle.load(open("final_tests/NN/dtextract/sick/test_nn_sick.p", "rb"), encoding='latin1')
#print(len(complex_models))
target_column = 28


for file in files:
    nominal_features = []
    df = pd.read_csv("./datasets/" + file + ".csv", header=0)
    if file == "breast":
        df = df.drop("id", 1)
    #if file == "voting":
        #df = df.drop("physician_fee_freeze", 1)
    if file == "nba_logreg":
        df = df.drop("Name", 1)
        df =df.fillna(0)
    y = df.target
    feature_names_orig = []
    for col in df.columns:
        if col != 'target':
            feature_names_orig.append(col)
            #if file == "voting":
            #    nominal_features.append(col)
    # if file == "voting":
        # nominal_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    if file == "heart":
        nominal_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        # nominal_features = [1, 2, 5, 6, 8, 10, 11, 12]
    #df_dummy = pd.get_dummies(df, columns=nominal_features)
    #y_dummy = df_dummy.target
    #df_dummy = df_dummy.drop("target", 1)
    #df = df.drop("target", 1)
    # print(feature_names)
    k = floor(sqrt(len(df.index)))
    n_neighbors = k if k % 2 == 1 else k-1
    for depth in depths:
        print(file, depth)
        boosted_unexplainable_avg = []
        boosted_ratio = []
        accuracy_boosted = []
        accuracy_complex = []
        accuracy_original = []
        accuracy_original_limited = []
        fidelity_original = []
        fidelity_original_limited = []
        fidelity_boosted = []
        random_fidelity = []
        conf_fidelity = []
        conf_avg_fidelity = []
        conf_leaf_fidelity = []
        max_fidelity = 0
        max_fidelity_class_weight = 0
        number_of_nodes_original_tree = []
        number_of_nodes_boosted_tree = []
        original_tree_depth = []
        distance_measure = []
        class0_boosting_factor = []
        class1_boosting_fator = []
        class0_generation_factor = []
        class1_generation_factor = []
        boosted_confidence_avg = []
        average_confidence = []

        for trial in trials:
            boosted_confidence = []
            print(trial)
            training_set = training_sets[trial]
            #print(training_set[28])
            # print(training_set.columns)
            test_set = test_sets[trial]
            #print(training_set)
            complex_model = complex_models[trial]
            y_train_dummy = training_set[target_column]
            trainDf_dummy = training_set.drop(target_column, 1)
            y_test_dummy = test_set[target_column]
            testDf_dummy = test_set.drop(target_column, 1)
            #nominal_features = ['1_0', '1_1', '1_nan', '2_0', '2_1', '2_2', '2_3', '2_nan', '5_0', '5_1', '5_nan',
            #                    '6_0', '6_1', '6_2', '6_nan', '8_0', '8_1', '8_nan', '10_0', '10_1',  '10_2', '10_nan',
            #                    '11_0', '11_1', '11_2', '11_3', '11_4', '11_nan', '12_0', '12_1', '12_2', '12_3',
            #                   '12_nan']
            #for col in trainDf_dummy.columns:
            #    nominal_features.append(col)
            #trainDf_dummy, testDf_dummy, y_train_dummy, y_test_dummy = train_test_split(df_dummy, y_dummy, test_size=0.1)
            #nominal_features = []

            nominal_features = ['1_F', '1_M', '1_nan', '2_0', '2_1', '2_nan', '3_0', '3_1', '3_nan', '4_0', '4_1',
                                '4_nan', '5_0', '5_1', '5_nan', '6_0', '6_1', '6_nan', '7_0', '7_1', '7_nan', '8_0',
                                '8_1', '8_nan', '9_0', '9_1', '9_nan', '10_0', '10_1', '10_nan', '11_0', '11_1',
                                '11_nan', '12_0', '12_1', '12_nan', '13_0', '13_1', '13_nan', '14_0', '14_1', '14_nan',
                                '15_0', '15_1', '15_nan', '16_0', '16_1', '16_nan', '18_0', '18_1', '18_nan', '20_0',
                                '20_1', '20_nan', '22_0', '22_1', '22_nan', '24_0', '24_1', '24_nan', '26_0', '26_nan',
                                '27_STMW', '27_SVHC', '27_SVHD', '27_SVI', '27_other', '27_nan']
            """               
            nominal_features = ['2_F', '2_M', '2_nan', '3_0', '3_1', '3_nan', '4_0', '4_1', '4_nan', '5_0', '5_1',
                                '5_nan', '6_0', '6_1', '6_nan', '7_0', '7_1', '7_nan', '8_0', '8_1', '8_nan', '9_0',
                                '9_1', '9_nan', '10_0', '10_1', '10_nan', '11_0', '11_1', '11_nan', '12_0', '12_1',
                                '12_nan', '13_0', '13_1', '13_nan', '14_0', '14_1', '14_nan', '16_0', '16_1', '16_nan',
                                '18_0', '18_1', '18_nan', '20_0', '20_1', '20_nan', '22_0', '22_1', '22_nan', '24_0',
                                '24_1', '24_nan']
                                """

            #for col in trainDf_dummy.columns:
            #    if file == "voting":
            #        nominal_features.append(col)
            train_dataset = pd.concat([trainDf_dummy, y_train_dummy], axis=1, sort=False)
            feature_names = []
            for col in trainDf_dummy.columns:
                if col != target_column:
                    feature_names.append(col)
            ranges = compute_features_range(trainDf_dummy, feature_names, nominal_features)
            #complex_model = RandomForestClassifier(n_estimators=1000)
            #complex_model.fit(trainDf_dummy, y_train_dummy)
            """
            if file == "voting":
                pass
                # complex_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5, 1), random_state=1) # BREAST
                #complex_model = RandomForestClassifier(n_estimators=1000)
                # complex_model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500,)))  # voting
            if file == "diabetes" or file == "breast32" or file == "heart":
                pass
                # complex_model = RandomForestClassifier(n_estimators=1000)
                complex_model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500,))
            complex_model.fit(pd.get_dummies(X_train, columns=nominal_features), y_train)
            """
            y_pred = complex_model.predict(testDf_dummy)
            #print(testDf_dummy)
            #print(y_pred)
            y_train_pred = complex_model.predict(trainDf_dummy)
            print("Training accuracy: ", accuracy_score(y_train_dummy, complex_model.predict(trainDf_dummy)))
            print(accuracy_score(y_test_dummy, y_pred))
            original_tree = tree.DecisionTreeClassifier()
            original_tree.fit(trainDf_dummy, y_train_dummy)
            original_tree_depth.append(original_tree.tree_.max_depth)
            #print(original_tree.tree_.value[0])
            y_original = original_tree.predict(testDf_dummy)
            original_tree_limited = tree.DecisionTreeClassifier(max_depth=depth)
            original_tree_limited.fit(trainDf_dummy, y_train_dummy)
            y_original_limited = original_tree_limited.predict(testDf_dummy)
            accuracy_complex.append(accuracy_score(y_test_dummy, y_pred))
            accuracy_original.append(accuracy_score(y_test_dummy, y_original))
            accuracy_original_limited.append(accuracy_score(y_test_dummy, y_original_limited))
            fidelity_original.append(accuracy_score(y_pred, y_original))
            fidelity_original_limited.append(accuracy_score(y_pred, y_original_limited))
            boosted_trees, features, boosting_factors, data_generation_factors = train_xboosted_trees(train_dataset, trainDf_dummy, y_train_dummy, class_weight,
                                                                           target_column,
                                                                           complex_model, depth, nominal_features,
                                                                           n_neighbors, testDf_dummy)
            #print("Trees attributes: ", boosted_trees[0].tree_.feature, boosted_trees[1].tree_.feature)
            boosted_trees_confidence = tree_confidence(boosted_trees, train_dataset, feature_names, target_column)
            #print(boosted_trees_confidence)

            number_of_nodes_original_tree.append(original_tree.tree_.node_count)
            number_of_nodes_boosted_tree.append(max(boosted_trees[0].tree_.node_count, boosted_trees[1].tree_.node_count))

            y_boosted = boosted_predict(boosted_trees, testDf_dummy, features, ranges, nominal_features, verbose=False)
            accuracy_boosted.append(accuracy_score(y_test_dummy, y_boosted))
            # print("Accuracy: ", accuracy_score(y_test, y_boosted))
            class0_boosting_factor.append(boosting_factors[0])
            class1_boosting_fator.append(boosting_factors[1])
            class0_generation_factor.append(data_generation_factors[0])
            class1_generation_factor.append(data_generation_factors[1])
            y_boosted_random = boosted_random_predict(boosted_trees, testDf_dummy)
            y_boosted_conf = boosted_confidence_predict(boosted_trees, testDf_dummy, feature_names, boosted_trees_confidence)
            y_boosted_avg_conf = boosted_sum_confidence_predict(boosted_trees, testDf_dummy, feature_names, boosted_trees_confidence)
            y_boosted_leaf_conf = boosted_leaf_confidence_predict(boosted_trees, testDf_dummy, feature_names, boosted_trees_confidence)
            fidelity_boosted.append(accuracy_score(y_pred, y_boosted))
            random_fidelity.append(accuracy_score(y_pred, y_boosted_random))
            conf_fidelity.append(accuracy_score(y_pred, y_boosted_conf))
            conf_avg_fidelity.append(accuracy_score(y_pred, y_boosted_avg_conf))
            conf_leaf_fidelity.append(accuracy_score(y_pred, y_boosted_leaf_conf))
            print("Fidelity: ", accuracy_score(y_pred, y_boosted))
            print("Random Fidelity: ", accuracy_score(y_pred, y_boosted_random))
            print("Confidence Fidelity: ", accuracy_score(y_pred, y_boosted_conf))
            print("Confidence Avg Fidelity: ", accuracy_score(y_pred, y_boosted_avg_conf))
            print("Confidence Leaf Fidelity: ", accuracy_score(y_pred, y_boosted_leaf_conf))
            print("CART Fidelity: ", accuracy_score(y_pred, y_original))
            print("CART Limited fidelity: ", accuracy_score(y_pred, y_original_limited))

            if accuracy_score(y_pred, y_boosted) > max_fidelity:
                max_fidelity = accuracy_score(y_pred, y_boosted)
                max_fidelity_class_weight = class_weight

            y_boosted0 = boosted_trees[0].predict(testDf_dummy)
            y_boosted1 = boosted_trees[1].predict(testDf_dummy)
            distance_count = 0
            boosted_unexplainable = 0
            for i in range(len(y_pred)):
                if y_pred[i] != y_boosted0[i] and y_pred[i] != y_boosted1[i]:
                    boosted_unexplainable += 1
                if y_boosted0[i] != y_boosted1[i]:
                    distance_count += 1

                path0 = boosted_trees[0].decision_path([testDf_dummy.iloc[i]])
                path1 = boosted_trees[1].decision_path([testDf_dummy.iloc[i]])
                prediction_path0 = path0.indices
                prediction_path1 = path1.indices
                max_confidence_over_path = 0
                confident_tree = None
                average_conf_tree_0 = []
                average_conf_tree_1 = []
                for item0 in prediction_path0:
                    class_tree = 0
                    node_ratio = get_confidence(boosted_trees_confidence, item0, y_pred[i], boosted_trees[0], feature_names, testDf_dummy.iloc[i], class_tree)
                    average_conf_tree_0.append(node_ratio)
                    if node_ratio > max_confidence_over_path:
                        max_confidence_over_path = node_ratio
                        confident_tree = class_tree
                conf_tree0 = mean(average_conf_tree_0)
                for item1 in prediction_path1:
                    class_tree = 1
                    node_ratio = get_confidence(boosted_trees_confidence, item1, y_pred[i], boosted_trees[1], feature_names, testDf_dummy.iloc[i], class_tree)
                    average_conf_tree_1.append(node_ratio)
                    if node_ratio > max_confidence_over_path:
                        max_confidence_over_path = node_ratio
                        confident_tree = class_tree
                conf_tree1 = mean(average_conf_tree_1)
                boosted_confidence.append(max_confidence_over_path)
                average_confidence.append(max(conf_tree0, conf_tree1))

            print("Distance count: ", distance_count)
            print("Number of test instances: ", len(y_boosted1))
            boosted_unexplainable_avg.append(boosted_unexplainable)
            distance_measure.append(distance_count)
            boosted_ratio.append((len(y_test_dummy) - boosted_unexplainable) / len(y_test_dummy))
            boosted_confidence_avg.append(mean(boosted_confidence))
            # print(boosted_confidence_avg)
            # print("Coverage: ", (len(y_test) - boosted_unexplainable) / len(y_test))
        output_file = open("final_tests/NN/2xAI/sick/param105.txt", 'a')
        output_file.writelines("Dataset: {}, \nTree depth: {},\nResults: \nUnexplainable: {} \nCoverage: {} \n"
                               "Accuracy_boosted: {}\nAccuracy_complex: {}\nFidelity: {}\nRandom Fidelity: {}\n"
                               "Confidence Fidelity: {}\nAvg Confidence Fidelity: {}\nLeaf Confidence Fidelity: {}\n"
                               "Fidelity CART: {}\nFidelity CART limited: {}\n"
                               "Number of nodes in boosted tree: {} \nNumber of nodes in original tree: {} \nDepth Original: {}\n"
                               "Class 0 boosting factor: {}\nClass 0 data generation factor: {}\nClass 1 boosting factor: {}\n"
                               "Class 1 data generation factor: {}\nConfidence: {}\nAverage Confidence: {}\n"
                               .format(file, depth, mean(boosted_unexplainable_avg),
                                       mean(boosted_ratio), mean(accuracy_boosted), mean(accuracy_complex),
                                       mean(fidelity_boosted), mean(random_fidelity), mean(conf_fidelity), mean(conf_avg_fidelity),
                                       mean(conf_leaf_fidelity), mean(fidelity_original), mean(fidelity_original_limited),
                                       mean(number_of_nodes_boosted_tree), mean(number_of_nodes_original_tree), mean(original_tree_depth), mean(class0_boosting_factor),
                                       mean(class0_generation_factor), mean(class1_boosting_fator),
                                       mean(class1_generation_factor), mean(boosted_confidence_avg), mean(average_confidence)))

        output_file.close()
