from sklearn import tree
from sklearn.tree import _tree
from copy import deepcopy
import pandas as pd
from adasyn_optimal import adasyn
from first_step_cost_training import cost_sensitive_training
from sklearn.metrics import accuracy_score
import random
import first_step_training
from adasyn_first_step import adasyn_first_step
from adasyn_second_step import adasyn_second_step

def is_leaf_node(decision_tree, index):
    is_leaf = 0
    if decision_tree.tree_.children_left[index] == -1 and decision_tree.tree_.children_right[index] == -1:
        is_leaf = 1
    return is_leaf


def get_ratio(dtree, node, true_label):
    node_values = dtree.tree_.value[node][0]
    total_num_samples = sum(node_values)
    if not is_leaf_node(dtree, node):
        if true_label == 0:
            child_left = dtree.tree_.children_left[node]
            child_left_samples = dtree.tree_.value[child_left][0]
            return child_left_samples[0]/sum(child_left_samples)
        elif true_label == 1:
            child_right = dtree.tree_.children_right[node]
            child_right_samples = dtree.tree_.value[child_right][0]
            return child_right_samples[0]/sum(child_right_samples)

    node_ratio = 0
    for i in range(len(node_values)):
        if true_label == i:
            node_ratio = node_values[i]/total_num_samples
    return node_ratio


def get_ratio1(dtree, node, true_label, attribute, sample):
    node_values = dtree.tree_.value[node][0]
    threshold = dtree.tree_.threshold[node]
    total_num_samples = sum(node_values)
    if not is_leaf_node(dtree, node):
        sample_value = sample[attribute]
        if sample_value <= threshold:
            child_left = dtree.tree_.children_left[node]
            child_left_samples = dtree.tree_.value[child_left][0]
            if true_label == 0:
                return child_left_samples[0]/sum(child_left_samples)
            else:
                return child_left_samples[1]/sum(child_left_samples)
        else:
            child_right = dtree.tree_.children_right[node]
            child_right_samples = dtree.tree_.value[child_right][0]
            if true_label == 0:
                return child_right_samples[0]/sum(child_right_samples)
            else:
                return child_right_samples[1]/sum(child_right_samples)

    node_ratio = 0
    for i in range(len(node_values)):
        if true_label == i:
            node_ratio = node_values[i]/total_num_samples
    return node_ratio


def get_real_ratio(dtree, node, true_label, attribute, sample, train_dataset):
    node_values = dtree.tree_.value[node][0]
    threshold = dtree.tree_.threshold[node]
    total_num_samples = sum(node_values)
    if not is_leaf_node(dtree, node):
        sample_value = sample[attribute]
        if sample_value <= threshold:
            child_left = dtree.tree_.children_left[node]
            child_left_samples = dtree.tree_.value[child_left][0]
            if true_label == 0:
                return child_left_samples[0]/sum(child_left_samples)
            else:
                return child_left_samples[1]/sum(child_left_samples)
        else:
            child_right = dtree.tree_.children_right[node]
            child_right_samples = dtree.tree_.value[child_right][0]
            if true_label == 0:
                return child_right_samples[0]/sum(child_right_samples)
            else:
                return child_right_samples[1]/sum(child_right_samples)

    node_ratio = 0
    for i in range(len(node_values)):
        if true_label == i:
            node_ratio = node_values[i]/total_num_samples
    return node_ratio


def custom_confidence(dtree, dpath, label):
    sum_conf = 0
    node_conf = 0
    for l in dpath:
        node_conf = get_ratio(dtree, l, label)
        sum_conf += node_conf

    return sum_conf * node_conf, node_conf


def compute_features_range(x_train, features, nominal_features):
    ranges = {}
    for feature in features:
        if feature in nominal_features:
            ranges[feature] = len(x_train[feature].unique())
        else:
            maximum = x_train[feature].max()
            minimum = x_train[feature].min()
            ranges[feature] = maximum - minimum
    return ranges


def compute_distance(dtree, dpath, sample, attributes, nominal_features, ranges):
    # print(dtree.tree_.feature)
    features = [
        attributes[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in dtree.tree_.feature]
    distance = 0
    for node in dpath:
        if not(is_leaf_node(dtree, node)):
            attribute = features[node]
            threshold = dtree.tree_.threshold[node]
            if attribute not in nominal_features:
                distance += abs((sample[attribute] - threshold)/ranges[attribute])
            else:
                distance += 1/ranges[attribute]

    return distance


def print_decision_path(dtree, attributes, path, sample, predicted_class, conf_dict, class_tree):
    features = [
        attributes[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in dtree.tree_.feature]

    print("Decision path for sample \n{} \nis:".format(sample))
    indent = "  "
    depth = 0
    for node in path:
        attr = features[node]
        threshold = dtree.tree_.threshold[node]
        #ratio = get_ratio1(dtree, node, predicted_class, attr, sample)
        ratio = get_confidence(conf_dict, node, predicted_class, dtree, attributes, sample, class_tree)
        if not(is_leaf_node(dtree, node)):
            if sample[attr] > threshold:
                threshold_sign = ">"
            else:
                threshold_sign = "<="
            print("{} X[{}] = {} {} {}: \n {}Node confidence: {}".format(indent * depth, attr, sample[attr], threshold_sign, threshold, indent * depth, ratio))
        else:
            prediction = predicted_class
            print("{}Final decision: class {} with confidence: {}\n".format(indent * depth, prediction, ratio))
        depth += 1

    return


def train_xboosted_trees_cost(data, xtrain, ytrain, boost_coef, target_column, model_to_explain, depth, nominal, n_neighbors, x_test):
    feature_names = []
    for col in data.columns:
        if col != target_column:
            feature_names.append(col)
    boosting_factors = cost_sensitive_training(xtrain, ytrain, boost_coef, depth)
    train_dataset = pd.concat([xtrain, ytrain], axis=1, sort=False)
    classes = train_dataset[target_column].unique()
    classes.sort()
    xboosted_trees = {}
    data_generation_factors = {}
    for class_label in classes:
        new_training_datsets = []
        boosted_dataset = train_dataset
        fidelity = []
        for i in range(boost_coef):
            new_train = adasyn(xtrain, ytrain, target_column, class_label, model_to_explain, nominal, n_neighbors, i)
            boosted_training = pd.concat([boosted_dataset, new_train], axis=0, sort=False)
            new_training_datsets.append(boosted_training)
            class_boost_ytrain = boosted_training[target_column]
            class_boost_xtrain = boosted_training.drop(target_column, 1)
            class_boosted_tree = tree.DecisionTreeClassifier(max_depth=depth, class_weight=boosting_factors[class_label])
            class_boosted_tree.fit(class_boost_xtrain, class_boost_ytrain.astype('int'))
            if i > 1:
                y_complex = model_to_explain.predict(new_train.drop(target_column, 1))
                fidelity.append(accuracy_score(y_complex, class_boosted_tree.predict(new_train.drop(target_column, 1))))
            else:
                fidelity.append(0)
        print(fidelity)
        print(fidelity.index(max(fidelity)))
        final_training_set = new_training_datsets[fidelity.index(max(fidelity))]
        data_generation_factors[class_label] = fidelity.index(max(fidelity))
        final_class_boosted_tree = tree.DecisionTreeClassifier(max_depth=depth, class_weight=boosting_factors[class_label])
        final_y = final_training_set[target_column]
        final_x = final_training_set.drop(target_column, 1)
        final_class_boosted_tree.fit(final_x, final_y.astype('int'))
        copy = deepcopy(final_class_boosted_tree)
        xboosted_trees[class_label] = copy

    return xboosted_trees, feature_names, boosting_factors, data_generation_factors


def train_xboosted_trees(data, xtrain_dummy, ytrain_dummy, boost_coef, target_column, model_to_explain, depth, nominal, n_neighbors, x_test):
    feature_names = []
    for col in data.columns:
        if col != target_column:
            feature_names.append(col)
    boosted_datasets, boosting_factors = first_step_training.cost_sensitive_training(xtrain_dummy, ytrain_dummy, target_column, 10, depth, nominal)
    train_dataset = pd.concat([xtrain_dummy, ytrain_dummy], axis=1, sort=False)
    classes = train_dataset[target_column].unique()
    classes.sort()
    xboosted_trees = {}
    data_generation_factors = {}
    for class_label in classes:
        new_training_datsets = []
        boosted_dataset = boosted_datasets[class_label]
        Minority_per_xi, Gi, most_common, m = adasyn_first_step(xtrain_dummy, ytrain_dummy, target_column, class_label, nominal, n_neighbors, 2)
        fidelity = []
        new_train = pd.DataFrame()
        for i in range(boost_coef):
            print(class_label, i)
            new_train = pd.concat([new_train, adasyn_second_step(xtrain_dummy, ytrain_dummy, target_column, m, most_common, model_to_explain, nominal, Gi, Minority_per_xi)], axis=0, sort=False)
            boosted_training = pd.concat([boosted_dataset, new_train], axis=0, sort=False)
            new_training_datsets.append(boosted_training)
            class_boost_ytrain = boosted_training[target_column]
            class_boost_xtrain = boosted_training.drop(target_column, 1)
            class_boosted_tree = tree.DecisionTreeClassifier(max_depth=depth)
            class_boosted_tree.fit(class_boost_xtrain, class_boost_ytrain.astype('int'))
            y_complex = new_train[target_column]
            # condition on tp and tn
            #TODO
            fidelity.append(accuracy_score(y_complex, class_boosted_tree.predict(new_train.drop(target_column, 1))))
        print(fidelity)
        print(fidelity.index(max(fidelity)))
        final_training_set = new_training_datsets[fidelity.index(max(fidelity))]
        data_generation_factors[class_label] = fidelity.index(max(fidelity))
        final_class_boosted_tree = tree.DecisionTreeClassifier(max_depth=depth)
        final_y = final_training_set[target_column]
        final_x = final_training_set.drop(target_column, 1)
        final_class_boosted_tree.fit(final_x, final_y.astype('int'))
        copy = deepcopy(final_class_boosted_tree)
        xboosted_trees[class_label] = copy

    return xboosted_trees, feature_names, boosting_factors, data_generation_factors


def boosted_predict(xboosted_trees, x_test, feature_names, ranges, nominal_features, verbose=False):
    y_predict = []
    for i in range(len(x_test)):
        predict_dict = {}
        for key, value in xboosted_trees.items():
            decision_path = value.decision_path([x_test.iloc[i]]).indices
            y_pred = int(value.predict([x_test.iloc[i]]))
            #if verbose:
            #    print_decision_path(value, feature_names, decision_path, x_test.iloc[i], y_pred)
            # confidence, leaf = custom_confidence(value, decision_path, y_pred)
            distance = compute_distance(value, decision_path, x_test.iloc[i], feature_names, nominal_features, ranges)
            if y_pred in predict_dict:
                predict_dict[y_pred] += distance
            else:
                predict_dict[y_pred] = distance
        predicted_class = 0
        max_confidence = 0

        for key, value in predict_dict.items():
            if value >= max_confidence:
                max_confidence = value
                predicted_class = key

        y_predict.append(predicted_class)

    return y_predict


def boosted_confidence_predict(xboosted_trees, x_test, feature_names, conf_dict, verbose=False):
    y_predict = []
    for i in range(len(x_test)):
        predict_dict = {}
        for key, value in xboosted_trees.items():
            decision_path = value.decision_path([x_test.iloc[i]]).indices
            y_pred = int(value.predict([x_test.iloc[i]]))
            #if verbose:
            #    print_decision_path(value, feature_names, decision_path, x_test.iloc[i], y_pred)
            # confidence, leaf = custom_confidence(value, decision_path, y_pred)
            max_confidence = 0
            for node in decision_path:
                node_confidence = get_confidence(conf_dict, node, y_pred, value, feature_names, x_test.iloc[i], key)
                if node_confidence >= max_confidence:
                    max_confidence = node_confidence
            predict_dict[y_pred] = max_confidence

        predicted_class = 0
        final_confidence = 0
        for key, value in predict_dict.items():
            if value >= final_confidence:
                final_confidence = value
                predicted_class = key

        y_predict.append(predicted_class)

    return y_predict


def boosted_sum_confidence_predict(xboosted_trees, x_test, feature_names, conf_dict, verbose=False):
    y_predict = []
    for i in range(len(x_test)):
        predict_dict = {}
        for key, value in xboosted_trees.items():
            decision_path = value.decision_path([x_test.iloc[i]]).indices
            y_pred = int(value.predict([x_test.iloc[i]]))
            if verbose:
                #print_decision_path(value, feature_names, decision_path, x_test.iloc[i], y_pred)
                print_decision_path(value, feature_names, decision_path, x_test.iloc[i], y_pred, conf_dict, key)
            # confidence, leaf = custom_confidence(value, decision_path, y_pred)
            total_confidence = 0
            for node in decision_path:
                node_confidence = get_confidence(conf_dict, node, y_pred, value, feature_names, x_test.iloc[i], key)
                total_confidence += node_confidence
            predict_dict[y_pred] = total_confidence / len(decision_path)

        predicted_class = 0
        final_confidence = 0
        for key, value in predict_dict.items():
            if value >= final_confidence:
                final_confidence = value
                predicted_class = key

        y_predict.append(predicted_class)

    return y_predict


def boosted_leaf_confidence_predict(xboosted_trees, x_test, feature_names, conf_dict, verbose=False):
    y_predict = []
    for i in range(len(x_test)):
        predict_dict = {}
        for key, value in xboosted_trees.items():
            decision_path = value.decision_path([x_test.iloc[i]]).indices
            y_pred = int(value.predict([x_test.iloc[i]]))
            total_confidence = 0
            node_confidence = 0.5
            for node in decision_path:
                if is_leaf_node(value, node):
                    node_confidence = get_confidence(conf_dict, node, y_pred, value, feature_names, x_test.iloc[i], key)
            predict_dict[y_pred] = node_confidence

        predicted_class = 0
        final_confidence = 0
        for key, value in predict_dict.items():
            if value >= final_confidence:
                final_confidence = value
                predicted_class = key

        y_predict.append(predicted_class)

    return y_predict


def boosted_random_predict(xboosted_trees, x_test):
    y_predict = []
    for i in range(len(x_test)):
        y_pred = [xboosted_trees[0].predict([x_test.iloc[i]]), xboosted_trees[1].predict([x_test.iloc[i]])]
        y_predict.append(random.choice(y_pred))

    return y_predict


def iterate(conf_dict, dataset, node, dtree, class_label, features, target_column):
    if not(is_leaf_node(dtree, node)):
        conf_dict[class_label][node] = {}
        attr = features[node]
        threshold = dtree.tree_.threshold[node]
        subset_left = dataset[attr] <= threshold
        left_train_dataset = dataset[subset_left]
        n1 = int(sum(left_train_dataset[target_column]))
        n0 = len(left_train_dataset[target_column]) - n1
        if n0 > 0 or n1 > 0:
            conf_dict[class_label][node]["L"] = [n0 / (n0 + n1), n1 / (n0 + n1)]
            iterate(conf_dict, left_train_dataset, dtree.tree_.children_left[node], dtree, class_label, features, target_column)
        else:
            child_left_values = dtree.tree_.value[dtree.tree_.children_left[node]][0]
            total_num = sum(child_left_values)
            conf_dict[class_label][node]["L"] = [child_left_values[0] / total_num, child_left_values[1] / total_num]
            iterate(conf_dict, left_train_dataset, dtree.tree_.children_left[node], dtree, class_label, features,
                    target_column)
        subset_right = dataset[attr] > threshold
        right_train_dataset = dataset[subset_right]
        n1 = int(sum(right_train_dataset[target_column]))
        n0 = len(right_train_dataset[target_column]) - n1
        if n1 > 0 or n0 > 0:
            conf_dict[class_label][node]["R"] = [n0 / (n0 + n1), n1 / (n0 + n1)]
            iterate(conf_dict, right_train_dataset, dtree.tree_.children_right[node], dtree, class_label, features, target_column)
        else:
            child_right_values = dtree.tree_.value[dtree.tree_.children_right[node]][0]
            total_num = sum(child_right_values)
            conf_dict[class_label][node]["R"] = [child_right_values[0] / total_num, child_right_values[1] / total_num]
            iterate(conf_dict, left_train_dataset, dtree.tree_.children_right[node], dtree, class_label, features,
                    target_column)
    else:
        n1 = int(sum(dataset[target_column]))
        n0 = len(dataset[target_column]) - n1
        if n0 > 0 or n1 > 0:
            conf_dict[class_label][node] = [n0 / (n0 + n1), n1 / (n0 + n1)]
        else:
            leaf_values = dtree.tree_.value[node][0]
            total_num = sum(leaf_values)
            conf_dict[class_label][node] = [leaf_values[0] / total_num, leaf_values[1] / total_num]


def tree_confidence(boosted_trees, train_dataset, attributes, target_column):
    confidence = {}
    for class_label, dtree in boosted_trees.items():
        confidence[class_label] = {}
        features = [
            attributes[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in dtree.tree_.feature]
        node = 0
        iterate(confidence, train_dataset, node, dtree, class_label, features, target_column)

    return confidence


def get_confidence(conf_dict, node, true_class, dtree, attributes, sample, class_tree):
    features = [
        attributes[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in dtree.tree_.feature]
    if not(is_leaf_node(dtree, node)):
        attr = features[node]
        threshold = dtree.tree_.threshold[node]
        if sample[attr] <= threshold:
            return max(conf_dict[class_tree][node]["L"])
        else:
            return max(conf_dict[class_tree][node]["R"])
    else:
        return conf_dict[class_tree][node][true_class]


