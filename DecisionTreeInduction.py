import pandas as pd
import math
from collections import Counter
import numpy as np
from Node import Node
from pprint import pprint
from ucimlrepo import fetch_ucirepo

def entropy(data):
    cnt = Counter(data) # count the number of each instance
    p = [x / len(data) for x in cnt.values()]

    return -sum([pi * math.log(pi) for pi in p]) # Info(data) = -sum( pi*log(pi) )

def info_gain(d: pd.DataFrame, attr, target_attr):
    # Split Data by Possible Vals of Attribute:
    d_split = d.groupby(attr)

    # Calculate Entropy for Target Attribute, as well as Proportion of Obs in Each Data-Split:
    entropies = []
    ratio = []

    for group_name, group_data in d_split:
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
    new_entropy = sum(entropy_ratio_dataframe['entropy'] * entropy_ratio_dataframe['|dj|/|d|'])
    old_entropy = entropy(d[target_attr])
    return old_entropy - new_entropy


#   generate decision tree
def generate_DT(D: pd.DataFrame, attr_list, target_attr, majority=None):
    N = Node(data=None, children={})
    classlist_in_d = Counter(D)

    #   if dataset tuple is entirely of a single class
    if len(classlist_in_d) == 1:
        return N.set(attr_list[0])
    #   if attribute list is empty (empty when all but the class that this ruleset is predicting)
    if D.empty or not attr_list:
        return N.set(majority)

    #   attribute selection
    gains = [info_gain(D, attr, target_attr) for attr in attr_list]
    index_of_max = gains.index(max(gains))  # pick one with highest gain
    split_criteria = attr_list[index_of_max]    # best splitting decision
    print(f"best attribute: {split_criteria}")
    N.set(split_criteria)

    max_key = max(classlist_in_d, key=classlist_in_d.get)
    majority_class = attr_list[max_key]
    new_attr_list = attr_list.remove(split_criteria)


    for attr, Dj in D.groupby(split_criteria):
        print(f"{attr}: {Dj}")
        N.append(generate_DT(Dj, new_attr_list, target_attr, majority_class))
    
    return N



########
df_shroom = pd.read_csv('data/mushroom_data.csv')

# Get Predictor Names (all but 'class')
attribute_names = list(df_shroom.columns)
attribute_names.remove('class')
print(df_shroom['class'])

dtree = generate_DT(df_shroom, attribute_names, "class")
pprint(dtree)