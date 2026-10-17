from naive_bayes_classifier.naivebayes import load_data, step_one, step_two, step_three

def main() -> None:
    train, test = load_data()

    class_counts, feature_value_counts = step_one(train)

    class_prior_probability, feature_value_likelihood = step_two(train, class_counts, feature_value_counts)

    step_three(test, train, class_counts, class_prior_probability, feature_value_likelihood)
