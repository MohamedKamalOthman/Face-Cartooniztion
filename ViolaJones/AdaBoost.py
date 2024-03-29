import numpy as np
import math
from weakclassifier import WeakClassifier
from progress.bar import Bar

# Understood
class AdaBoost:
    def __init__(self, T=10) -> None:
        """
        T is the number of WeakClassifiers that should be used
        """
        self.T = T
        self.alphas = []
        self.clfs = []

    # Understood
    def train(self, features_values, labels, features):
        """
        features_values: List of features values
        labels: images label 0-> Non-face  1->Face
        features
        """
        pos_num = np.sum(labels)
        neg_num = len(labels) - pos_num
        weights = np.zeros(len(labels)).astype(np.float32)

        # Initialize weights according to paper equations
        for i in range(len(labels)):
            if labels[i] == 1:
                weights[i] = 1.0 / (2.0 * pos_num)
            else:
                weights[i] = 1.0 / (2.0 * neg_num)

        print("Training .......")
        for t in range(self.T):
            print("Training %d classifiers out of %d" % (t + 1, self.T))

            # Normalizing Weights
            w_sum = np.sum(weights)
            if w_sum == 0.0:
                print("Exiting Weights are zero .... Check Training set")
                break
            weights = weights / w_sum

            # print("Training Weak Classifiers....")
            W_C = self.train_weak(features_values, labels, weights, features)
            # print("- Num. weak classifiers: " + str(len(W_C)))
            # Select best  classifier with min error
            # print("Selecting best weak classifier....")
            clf, error, incorrectness = self.select_best(
                W_C, features_values, labels, weights
            )
            if error <= 0.5:
                # Compute alpha, beta
                beta = error / (1.0 - error)
                alpha = math.log(1.0 / (beta + 1e-18))

                # Update weights
                weights = np.multiply(weights, beta ** (1 - incorrectness))

                # Save parameters
                self.alphas.append(alpha)
                self.clfs.append(clf)

    # Understood
    def train_weak(self, features_values, labels, weights, features):
        """
        Finds the optimal thresholds for each weak classifier given the current weights
          Args:
            features_values: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            labels: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
            weights: A numpy array of shape len(training_data). The ith element is the weight assigned to the ith training example
        """
        classifiers = []
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, labels):
            if label == 1:
                total_pos += w
            else:
                total_neg += w
        bar = Bar(
            "Processing Weak Classifiers",
            max=len(features_values),
            suffix="%(percent)d%% - %(elapsed_td)s - %(eta_td)s",
        )
        for i in bar.iter(range(len(features_values))):
            clf = WeakClassifier(haar_feature=features[i])
            clf.train(features_values[i], labels, weights, total_pos, total_neg)
            classifiers.append(clf)
        bar.finish()
        return classifiers

    # Understood
    def select_best(self, weak_clfs, features_values, labels, weights):
        """
        Selects the best weak classifier for the given weights
        weak_clfs : An array of weak classifiers
        weights: An array of weights corresponding to each training example
        """
        best_clf, min_error, best_accuracy = None, float("inf"), None
        i = -1
        for clf in weak_clfs:
            i += 1
            incorrectness = np.abs(
                clf.classify_with_feature(features_values[i]) - labels
            )
            error = float(np.sum(np.multiply(incorrectness, weights))) / len(
                incorrectness
            )

            if error < min_error:
                best_clf, min_error, best_accuracy = clf, error, incorrectness

        return best_clf, min_error, best_accuracy

    # Understood
    def classify(self, integral_img, scale=1.0):
        total = sum(
            list(
                map(
                    lambda z: z[0] * z[1].classify(integral_img, scale),
                    zip(self.alphas, self.clfs),
                )
            )
        )  # Weak classifiers
        return 1 if total >= 0.5 * sum(self.alphas) else 0
