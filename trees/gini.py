from collections import defaultdict

# MODIFIED FROM: https://search.brave.com/search?q=gini+impurity+calculation+python+without+numpy&source=web&conversation=346af53a924381d40c1ec1&summary=1
def gini_impurity(data) -> float:
    """
    Calculates the GINI impurity.  The last column must be the class label.

    0 indicates a pure score, and 1 indicates high impurity
    """
    # Calculate the total number of instances
    total_rows = len(data)

    # Calculate the number of instances for each class
    class_counts = defaultdict(int)
    for instance in data:
        label = instance[-1]  # Assuming the last element is the class label
        class_counts[label] += 1

    # Calculate the Gini Impurity
    gini = 1
    gini -= sum([(count/total_rows)**2 for count in class_counts.values()])

    return gini

if __name__ == '__main__':
    # Example usage
    data = [
        [1, 0, 'yes'],
        [1, 1, 'yes'],
        [0, 1, 'no'],
        [0, 0, 'no'],
        [1, 0, 'no']
    ]

    print(f"Gini Impurity: {gini_impurity(data)}")