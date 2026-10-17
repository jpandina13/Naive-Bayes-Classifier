import pandas
import numpy as np
import sys
from pathlib import Path

ALPHA = 0.1

def load_data():
    """Load data from CSV and split it between train and test datasets.

    The 'train' dataset will contain 70% of the tuples, while the 'test' 
    dataset will contain the other 30%.

    Returns:
        tuple (dataframe, dataframe): the train dataset and the test dataset are returned together as a tuple.
    """
    data = pandas.read_csv(Path(__file__).parent.parent.parent / "data/zoo.csv")
    train = data.sample(frac=0.7)
    test = data.drop(train.index)
    return (train, test)


def step_one(train): # Rename to calculate_counts
    """Calculate the class_counts and the feature_value_counts of the training dataset.

    The class_counts dictionary will contain the number of occurences for each class. The
    feature_value_counts dictionary will contain the number of specific feature values that 
    occurred within each class.

    Args:
        train (dataframe): the training dataset.

    Returns:
        tuple (dict, dict): the class_counts and feature_value_counts dictionaries.
    """
    # This section successfully loads the counts of the train dataset into these dictionaries
    # feature_value_counts = {classification: {feature_name: {feature_value: count}}}
    class_counts = {}
    feature_value_counts = {}
    for index, row in train.iterrows():
        classification = row['class_type']
        if classification not in class_counts:
            class_counts[classification] = 1
        else:
            class_counts[classification] += 1
        if classification not in feature_value_counts:
            feature_value_counts[classification] = {}
        for featureIndex, featureValue in row.items():
            if featureIndex not in feature_value_counts[classification]:
                feature_value_counts[classification][featureIndex] = {}
            if featureValue not in feature_value_counts[classification][featureIndex]:
                feature_value_counts[classification][featureIndex][featureValue] = 1
            else:
                feature_value_counts[classification][featureIndex][featureValue] += 1
    return (class_counts, feature_value_counts)


def step_two(train, class_counts, feature_value_counts):
    """Calculate each class_prior_probability and each feature_value_likelihood
    """
    # This section calculates the classification prior probability along with the likelihoods of each feature value.
    # feature_value_likelihood = {classification: {feature_name: {feature_value: likelihood}}}
    numberOfInstances = len(train)
    class_prior_probability = {classification: (count / numberOfInstances) for classification, count in class_counts.items()}
    feature_value_likelihood = {}
    for classification in feature_value_counts:
        feature_value_likelihood[classification] = {}
        for feature_name in feature_value_counts[classification]:
            feature_value_likelihood[classification][feature_name] = {}
            for feature_value in feature_value_counts[classification][feature_name]:
                count = feature_value_counts[classification][feature_name][feature_value]
                feature_value_likelihood[classification][feature_name][feature_value] =(count + ALPHA) / (class_counts[classification] + ALPHA * train[feature_name].nunique())
    return (class_prior_probability, feature_value_likelihood)



def step_three(test, train, class_counts, class_prior_probability, feature_value_likelihood):
    """
    """
    # TESTING PHASE

    test['predicted'] = -1
    test['probability'] = -0.5
    test['correct?'] = 'unknown'
    totalCorrect = 0
    totalInstances = len(test)
    for index, row in test.iterrows():
        # Given a tuple, find the highest probability classification
        highest_probability = -np.inf
        best_classification = None
        for classification in class_prior_probability:
            
            posterior_prob = np.log2(class_prior_probability[classification])
            # np.log2 means log base 2

            for feature_name, value in row.items():
                if feature_name in ['class_type', 'predicted', 'probability', 'correct?']:
                    continue
                if value not in feature_value_likelihood[classification][feature_name]: # Feature value wasn't encountered during training
                    posterior_prob += np.log2(ALPHA / (class_counts[classification] + ALPHA * train[feature_name].nunique()))
                else:
                    posterior_prob += np.log2(feature_value_likelihood[classification][feature_name][value])
            if posterior_prob > highest_probability:
                highest_probability = posterior_prob
                best_classification = classification
        test.at[index, 'predicted'] = best_classification
        test.at[index, 'probability'] = np.power(2, highest_probability)
        if best_classification == row['class_type']:
            totalCorrect += 1
            test.at[index, 'correct?'] = 'CORRECT'
        else:
            test.at[index, 'correct?'] = 'wrong'

    accuracy = totalCorrect / totalInstances

    test.to_csv(sys.stdout, index=False)


def main():
    train, test = load_data()

    class_counts, feature_value_counts = step_one(train)

    class_prior_probability, feature_value_likelihood = step_two(train, class_counts, feature_value_counts)

    step_three(test, train, class_counts, class_prior_probability, feature_value_likelihood)


