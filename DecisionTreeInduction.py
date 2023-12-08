import pandas as pd
import math
from collections import Counter
import numpy as np
from Node import Node
from pprint import pprint
from ucimlrepo import fetch_ucirepo

def entropy(data):
    hashed_data = Counter(data) # dictionary of data, indiced by class
    entropy_list= []
    for x in hashed_data.values():
        p = (x / len(data))
        entropy_list.append(-p * math.log(p, 2))

    return sum(entropy_list) # Info(data) = -sum( pi*log(pi) )

def info_gain(d: pd.DataFrame, attr, target_attr):
    # Split Data by Possible Vals of Attribute:
    d_split = d.groupby(attr)
    pprint(d_split.indices)
    # Calculate Entropy for Target Attribute, as well as Proportion of Obs in Each Data-Split:
    entropies = []
    ratio = []

    for group_name, group_data in d_split:
        # print(group_data[target_attr])
        group_entropy = entropy(group_data[target_attr])    # log(|dj|/|d|)
        entropies.append(group_entropy) # Info(D)
        
        group_ratio = len(group_data) / len(d) # |dj|/|d|
        ratio.append(group_ratio)    # Info_a(D)

    # Create DataFrame with Entropy and Proportion of Observations:
    entropy_ratio_dataframe = pd.DataFrame({
        'entropy': entropies,
        '|dj|/|d|': ratio
    }, index=d_split.groups.keys())

    # Calculate Information Gain:
    expected_entropy = sum(entropy_ratio_dataframe['entropy'] * entropy_ratio_dataframe['|dj|/|d|'])
    original_entropy = entropy(d[target_attr])
    return original_entropy - expected_entropy

def generate_DT(D: pd.DataFrame, attr_list, target_attr, majority=None):
    # Tally target attribute:
    classlist_in_d = Counter(x for x in D[target_attr])

    # First check: Is this split of the dataset homogeneous?
    if len(classlist_in_d) == 1:
        return list(classlist_in_d.keys())[0]

    # Second check: Is this split of the dataset empty?
    # if yes, return a default value
    if D.empty or (not attr_list):
        return majority

    # Otherwise: This dataset is ready to be divvied up!
    # # Get Default Value for next recursive call of this function:
    index_of_max = list(classlist_in_d.values()).index(max(classlist_in_d.values()))
    majority_class = list(classlist_in_d.keys())[index_of_max]  # most common value of target attribute in dataset

    # Choose Best Attribute to split on:
    gains = [info_gain(D, attr, target_attr) for attr in attr_list]
    index_of_max = gains.index(max(gains))  # pick one with highest gain
    split_criteria = attr_list[index_of_max]    # best splitting decision

    # Create an empty tree, to be populated in a moment
    N = {split_criteria: {}}
    new_attr_list = []
    for i in attr_list:
        if i != split_criteria:
            new_attr_list.append(i)

    # recursion
    for attr, Dj in D.groupby(split_criteria):
        subtree = generate_DT(Dj, new_attr_list, target_attr, majority_class)
        N[split_criteria][attr] = subtree
    return N

def decision_tree_induction(D: pd.DataFrame, attr_list, target_attr):
    print("generating Decision Tree")
    targetattr_hashed = Counter(x for x in D[attributes_list])
    index_of_max = list(targetattr_hashed.values()).index(max(targetattr_hashed.values()))
    majority_class = list(targetattr_hashed.keys())[index_of_max]  # most common value of target attribute in dataset
    return generate_DT(D, attr_list, target_attr, majority_class)


########

id = 73
data = fetch_ucirepo(id=id)
dataset = data['data']['original']

# Get class to predict
predicting_class = "poisonous"
attributes_list = list(dataset.columns)
attributes_list.remove(predicting_class)

total_rows = int(dataset.shape[0] * .8)
training_data = dataset.iloc[1:total_rows]  # 80% of data as training data

dtree = decision_tree_induction(dataset, attributes_list, predicting_class)
pprint(dtree)