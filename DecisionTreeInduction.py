import pandas as pd
import math
from collections import Counter
import numpy as np
from pprint import pprint
from ucimlrepo import fetch_ucirepo
from memory_profiler import profile



def entropy(data):
    hashed_data = Counter(data) # dictionary of data, indiced by class
    entropy_list= []
    for x in hashed_data.values():
        p = (x / len(data))
        entropy_list.append(p * math.log(p, 2))

    return sum(entropy_list) # Info(data) = -sum( pi*log(pi) )
def info_gain(d: pd.DataFrame, attr, target_attr):
    # data by attribute
    d_attr = d.groupby(attr)
    # print(d_attr.indices)
    
    # |dj|/|d| ratio and entropy
    ratio = []
    entropies = []

    for i, j in d_attr:
        # print(group_data[target_attr])
        ratio.append(len(j) / len(d))    # |dj|/|d|
        entropies.append(entropy(j[target_attr])) # pi*log(pi)

    # Calculate Information Gain:
    expected_entropy = sum(np.array(ratio) * np.array(entropies))
    original_entropy = entropy(d[target_attr])
    gain = original_entropy - expected_entropy

    # split info to prevent overfitting
    split_info = entropy(entropies) # |dj|/|d| * log(|dj|/|d|)
    gain_ratio_inversed = split_info / gain  # accounting for dividing by zero
    return gain_ratio_inversed

# Selects the best attribute depending on information gain
def attribute_selection(D, attr_list, target_attr):
    gains = [info_gain(D, attr, target_attr) for attr in attr_list] # gain correspond with each attribute in the list
    index_of_max = gains.index(min(gains))  # to prevent divide-by-zero cases, gain is inversed
    split_criteria = attr_list[index_of_max]    # best class to split
    return split_criteria

# Generates decision tree from dataset (dataset has to be type Dataframe (fetching datasets from ucirepo))
@profile
def generate_DT(D: pd.DataFrame, attr_list, target_attr, majority=None):
    # count target tags in D
    classlist_in_d = Counter(x for x in D[target_attr])

    # if D is all one class
    if len(classlist_in_d) == 1:
        return list(classlist_in_d.keys())[0]

    # is D empty
    if D.empty or not attr_list:
        return majority

    index_of_max = list(classlist_in_d.values()).index(max(classlist_in_d.values()))
    majority_class = list(classlist_in_d.keys())[index_of_max]  # most common value of target attribute in dataset

    # split criteria
    split_criteria = attribute_selection(D, attr_list, target_attr)

    N = {split_criteria: {}}    # Initialize node using python Dictionary data structure

    #   pass the attribute list without the best attribute (greedy part)
    new_attr_list = []
    for i in attr_list:
        if i != split_criteria:
            new_attr_list.append(i)

    # get tree node from recursion
    for attr, Dj in D.groupby(split_criteria):
        if D.empty or not attr_list:
            N[split_criteria][attr] = majority_class            
        else:
            subtree = generate_DT(Dj, new_attr_list, target_attr, majority_class)
            N[split_criteria][attr] = subtree
    return N

# classifies using the dtree
# query is a type-Dictionary attributes of the instance being classified
@profile
def classify(query, dtree, default):
    node = list(dtree.keys())[0]
    attr_value = query[node]
    # find if the first attribute in decision tree is in the test line
    if attr_value in dtree[node].keys():
        result = dtree[node][attr_value]
        if isinstance(result, dict): # child could be another dictionary like the parent, then traverse
            return classify(query, result, default)
        else:
            return result
    else:
        return default
#######################
fetch_list = [19, 73, 105, 936, 827]
id = 19
fetched_data = fetch_ucirepo(id=id)

dataset_name = fetched_data.metadata.name
dataset = fetched_data.data.original
print(f"Dataset title: {dataset_name}")
# Get class to predict
predicting_class = list(fetched_data.data.targets)[0] # data to be predicted
attributes_list = list(fetched_data.data.features)    # attribute list

training_rows = int(dataset.shape[0] * .7)
training_data = dataset.iloc[1:training_rows]  # 70% of data as training data
test_data = dataset.iloc[training_rows:]   # 30% as test data

@profile
def decision_tree_induction(D: pd.DataFrame, attr_list, target_attr):
    print("Generating Decision Tree...")
    targetattr_hashed = Counter(x for x in D[attributes_list])
    index_of_max = list(targetattr_hashed.values()).index(max(targetattr_hashed.values()))
    majority_class = list(targetattr_hashed.keys())[index_of_max]  # most common value of target attribute in dataset
    return generate_DT(D, attr_list, target_attr, majority_class)    
dtree = decision_tree_induction(training_data, attributes_list, predicting_class)
print(f"{dataset_name} Decision tree:")
pprint(dtree)   # looks nicer to show tree

# default handling, get the class that is close-to-least present in the dataset (in this case we're using median).
# This is a band-aid solution to the fitting problem
predict_data_possibilities = list(Counter(dataset[predicting_class]))    # sorted list of data in the prediction class (the classification tags)
median_point = (int)(-1 + len(predict_data_possibilities) * 0.5) if len(predict_data_possibilities) > 2 else (int)(len(predict_data_possibilities)-1)
default_tag = predict_data_possibilities[median_point]

# passes each row from the database for classification
# for row_dict in test_data.to_dict(orient='records'):
#     classed = classify(row_dict, dtree, default_tag)
#     print(f"{row_dict} ==> {classed}")

# obtaining accuracy
# sum( the number of times the decision tree has yielded same classification as training_dataset / size of training_dataset
tested_classification = test_data.apply(classify, axis=1, args=(dtree, default_tag))
print(f'Accuracy of decision tree: {100 * sum(test_data[predicting_class]==tested_classification) / len(test_data.index)}%')