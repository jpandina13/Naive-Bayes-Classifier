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
    # feature_value_counts = {feature_name: {class_name: {feature_value: count}}}
    class_counts = {}
    feature_value_counts = {}
    for index, row in train.iterrows():
        target = row['class_type']
        if target not in class_counts:
            class_counts[target] = 1
        else:
            class_counts[target] += 1
        for featureIndex, featureValue in row.items():
            if featureIndex not in feature_value_counts:
                feature_value_counts[featureIndex] = {}
            if target not in feature_value_counts[featureIndex]:
                feature_value_counts[featureIndex][target] = {}
            if featureValue not in feature_value_counts[featureIndex][target]:
                feature_value_counts[featureIndex][target][featureValue] = 1
            else:
                feature_value_counts[featureIndex][target][featureValue] += 1
    return (class_counts, feature_value_counts)


def step_two(train, class_counts, feature_value_counts):
    # This section calculates the class prior probability along with the likelihoods of each feature value.
    # featureLikelihood = { feature_name : {class_name : {feature_value: likelihood}}}
    numberOfInstances = len(train)
    classPriorProbability = {target: (count / numberOfInstances) for target, count in class_counts.items()}
    featureLikelihood = {}
    for feature_name in feature_value_counts: 
        for target in feature_value_counts[feature_name]:
            for feature_value in feature_value_counts[feature_name][target]:
                count = feature_value_counts[feature_name][target][feature_value]
                if feature_name not in featureLikelihood:
                    featureLikelihood[feature_name] = {}
                if target not in featureLikelihood[feature_name]:
                    featureLikelihood[feature_name][target] = {}
                featureLikelihood[feature_name][target][feature_value] = (count + ALPHA) / (class_counts[target] + ALPHA * train[feature_name].nunique())
    return (classPriorProbability, featureLikelihood)

def step_three(test, train, class_counts, classPriorProbability, featureLikelihood):
    # TESTING PHASE
    # Note: you might encounter feature values that were not in the train set. Their likelihoods are ( 0 + alpha ) / ( class_counts[target] + alpha * data[feature_name].nunique() )
    test['predicted'] = -1
    test['probability'] = -0.5
    test['correct?'] = 'unknown'
    totalCorrect = 0
    totalInstances = len(test)
    for index, row in test.iterrows():
        max_log_prob = -np.inf
        best_class = None
        for target in classPriorProbability:
            posterior_prob = np.log2(classPriorProbability[target])
            for featureIndex, featureValue in row.items():
                if featureIndex == 'class_type' or featureIndex == 'predicted' or featureIndex == 'probability' or featureIndex == 'correct?':  
                    continue
                if featureValue not in featureLikelihood[featureIndex][target]:
                    #Feature value wasn't encountered during training
                    posterior_prob += np.log2(ALPHA / (class_counts[target] + ALPHA * train[feature_name].nunique()))
                else:
                    posterior_prob += np.log2(featureLikelihood[featureIndex][target][featureValue])
            if posterior_prob > max_log_prob:
                max_log_prob = posterior_prob
                best_class = target
        test.at[index, 'predicted'] = best_class
        test.at[index, 'probability'] = np.power(2, max_log_prob)
        if best_class == row['class_type']:
            totalCorrect += 1
            test.at[index, 'correct?'] = 'CORRECT'
        else:
            test.at[index, 'correct?'] = 'wrong'

    accuracy = totalCorrect / totalInstances

    test.to_csv(sys.stdout, index=False)


def main():
    train, test = load_data()

    class_counts, feature_value_counts = step_one(train)

    classPriorProbability, featureLikelihood = step_two(train, class_counts, feature_value_counts)

    step_three(test, train, class_counts, classPriorProbability, featureLikelihood)


