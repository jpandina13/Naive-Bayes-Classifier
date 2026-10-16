import pandas
import numpy as np
import sys

ALPHA = 0.1

def load_data():
    data = pandas.read_csv('zoo.csv')
    train = data.sample(frac=0.7)
    test = data.drop(train.index)
    return (train, test)


def step_one(train):
    # This section successfully loads the counts of the train dataset into these dictionaries
    # feature_value_counts = {feature_name: {classification: {feature_value: count}}}
    class_counts = {}
    feature_value_counts = {}
    for index, row in train.iterrows():
        classification = row['class_type']
        if classification not in class_counts:
            class_counts[classification] = 1
        else:
            class_counts[classification] += 1
        for featureIndex, featureValue in row.items():
            if featureIndex not in feature_value_counts:
                feature_value_counts[featureIndex] = {}
            if classification not in feature_value_counts[featureIndex]:
                feature_value_counts[featureIndex][classification] = {}
            if featureValue not in feature_value_counts[featureIndex][classification]:
                feature_value_counts[featureIndex][classification][featureValue] = 1
            else:
                feature_value_counts[featureIndex][classification][featureValue] += 1
    return (class_counts, feature_value_counts)


def step_two(train, class_counts, feature_value_counts):
    # This section calculates the classification prior probability along with the likelihoods of each feature value.
    # featureLikelihood = { feature_name : {classification : {feature_value: likelihood}}}
    numberOfInstances = len(train)
    class_prior_probability = {classification: (count / numberOfInstances) for classification, count in class_counts.items()}
    feature_value_likelihood = {}
    for feature_name in feature_value_counts: 
        for classification in feature_value_counts[feature_name]:
            for feature_value in feature_value_counts[feature_name][classification]:
                count = feature_value_counts[feature_name][classification][feature_value]
                if feature_name not in feature_value_likelihood:
                    feature_value_likelihood[feature_name] = {}
                if classification not in feature_value_likelihood[feature_name]:
                    feature_value_likelihood[feature_name][classification] = {}
                feature_value_likelihood[feature_name][classification][feature_value] = (count + ALPHA) / (class_counts[classification] + ALPHA * train[feature_name].nunique())
    return (class_prior_probability, feature_value_likelihood)

def step_three(test, train, class_counts, class_prior_probability, feature_value_likelihood):
    # TESTING PHASE
    # Note: you might encounter feature values that were not in the train set. Their likelihoods are ( 0 + alpha ) / ( class_counts[classification] + alpha * data[feature_name].nunique() )
    test['predicted'] = -1
    test['probability'] = -0.5
    test['correct?'] = 'unknown'
    totalCorrect = 0
    totalInstances = len(test)
    for index, row in test.iterrows():
        max_log_prob = -np.inf
        best_classification = None
        for classification in class_prior_probability:
            posterior_prob = np.log2(class_prior_probability[classification])
            for featureIndex, featureValue in row.items():
                if featureIndex == 'classification_type' or featureIndex == 'predicted' or featureIndex == 'probability' or featureIndex == 'correct?':  
                    continue
                if featureValue not in feature_value_likelihood[featureIndex][classification]:
                    #Feature value wasn't encountered during training
                    posterior_prob += np.log2(ALPHA / (class_counts[classification] + ALPHA * train[feature_name].nunique()))
                else:
                    posterior_prob += np.log2(feature_value_likelihood[featureIndex][classification][featureValue])
            if posterior_prob > max_log_prob:
                max_log_prob = posterior_prob
                best_classification = classification
        test.at[index, 'predicted'] = best_classification
        test.at[index, 'probability'] = np.power(2, max_log_prob)
        if best_classification == row['classification_type']:
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


