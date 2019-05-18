#!/usr/bin/python3

import math
import numpy as np
import cv2 as cv
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif

# Compute the integral image of an input image
def integral_img(img):

    ii = np.zeros(img.shape)
    s = np.zeros(img.shape)

    for y in range(len(img)):
        for x in range(len(img[y])):
            s[y][x] = s[y-1][x] + img[y][x] if y >= 1 else img[y][x]
            ii[y][x] = ii[y][x-1] + s[y][x] if x >= 1 else s[y][x]

    return ii


# Object to keep track of rectangular regions
class Rect:

    # Set x,y coordinates and width, height
    def __init__(self, x, y, w, h):

        self.x = x
        self.y = y
        self.w = w
        self.h = h

    # Compute sum of pixel intensities using integral image ii
    def compute_feature(self, ii):

        return ii[self.y+self.h][self.x+self.w] - ii[self.y+self.h][self.x] - ii[self.y][self.x+self.w] + ii[self.y][self.x]


# Weak classifier
class WeakClass:

    # For each weak classifier, there is a threshold and polarity associated
    # with it. We initialize the weak classifier with a set of positive regions,
    # a set of negative regions and the threshold/polarity values
    def __init__(self, regs_pos, regs_neg, threshold, polarity):

        self.regs_pos = regs_pos
        self.regs_neg = regs_neg
        self.threshold = threshold
        self.polarity = polarity

    # Viola-Jones weak classifier function
    def classify(self, int_img):

        # Calculate the features of the regions for this classifier
        feature = lambda ii: sum([p.compute_feature(ii) for p in
            self.regs_pos]) - sum([n.compute_feature(ii) for n in
                self.regs_neg])

        return 1 if self.polarity*feature(int_img) < self.polarity*self.threshold else 0


# Class for the Viola Jones algorithm
class ViolaJones:

    # Initialize algorithm with the desired number of weak classifiers T
    def __init__(self, T = 10):

        self.T = T
        self.alphas = []
        self.classifiers = []

    # Build list of ordered pairs of features ( [pos], [neg] )
    # This is converted to an interator in an attempt to save memory
    def build_features(self, img_shape):

        H, W = img_shape

        features = []

        # We use three particular Haar features and calculate the feature values
        # for every possible feature window over the image 
        for w in range(1, W+1):
            for h in range(1, H+1):
                i = 0
                while i + w < W:
                    j = 0
                    while j + h < H:
                        # Generate rectangular regions for computing Haar features
                        current = Rect(i, j, w, h)
                        right = Rect(i+w, j, w, h)
                        bottom = Rect(i, j+h, w, h)
                        right_2 = Rect(i+2*w, j, w, h)
                        bottom_2 = Rect(i, j+2*h, w, h)
                        bottom_right = Rect(i+w, j+h, w, h)

                        # Add Haar features if regions are within image
                        if i + 2*w < W:
                            features.append( ([right], [current]) )

                        if j + 2*h < H:
                            features.append( ([current], [bottom]) )

                        if i + 3*w < W:
                            features.append( ([right], [right_2, current]) )

                        if j + 3*h < H:
                            features.append( ([bottom], [bottom_2, current]) )

                        if i + 2*w < W and j + 2*h < H:
                            features.append( ([right, bottom], [current,
                                bottom_right]) )
                        j += 1
                    i += 1

        return features

    # Apply all features 
    def apply_features(self, features, train_data):

        print('Computing features...')

        applied_features = []

        # Compute features with integral image ii and positive & negative regions
        for reg_pos, reg_neg in features:

            feature_vals = []
            for data in train_data:
                ii = data[0]
                total_sum = 0
                for p in reg_pos:
                    total_sum += p.compute_feature(ii)
                for n in reg_neg:
                    total_sum -= n.compute_feature(ii)

                feature_vals.append(total_sum)

            applied_features.append(feature_vals)

        return np.array(applied_features)

    # Train weak classifiers
    def train_weak(self, applied_features, labels, features, weights):

        ptotal = ntotal = 0

        for w, label in zip(weights, labels):
            if label == 1:
                ptotal += w
            else:
                ntotal += w

        classifiers = []
        numfeatures = applied_features.shape[0]

        for index, feature in enumerate(applied_features):

            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d/%d classifiers" % (len(classifiers),
                    numfeatures), end='\r' )

            applied_feature = sorted(zip(weights, feature, labels), key=lambda x: x[1])
            pseen = nseen = 0
            pweights = nweights = 0
            min_error = math.inf
            best_feature = None
            best_threshold = None
            best_polarity = None

            for w, f, label in applied_feature:
                error = min(nweights + ptotal - pweights, pweights + ntotal - nweights)

                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pseen > nseen else -1

                if label == 1:
                    pseen += 1
                    pweights += w
                else:
                    nseen += 1
                    nweights += w

            cl = WeakClass(best_feature[0], best_feature[1],
                    best_threshold, best_polarity)
            classifiers.append(cl)

        print("Trained %d/%d classifiers" % (len(classifiers), numfeatures) )

        return classifiers

    # Find best weak classifier by calculating average weighted error
    def select_best(self, classifiers, weights, train_data):

        best_classifier = None
        best_error = math.inf
        best_accuracy = None

        for cl in classifiers:
            error = 0
            accuracy = []

            for data, w in zip(train_data, weights):
                correctness = abs(cl.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w*correctness

            error /= len(train_data)
            
            if error < best_error:
                best_classifier = cl
                best_error = error
                best_accuracy = accuracy

        return best_classifier, best_error, best_accuracy

    # Train classifier given training set and number of positive/negative
    # examples
    def train(self, training, n_pos, n_neg):

        print('Positive: ', n_pos)
        print('Negative: ', n_neg)

        weights = np.zeros(len(training))
        train_data = []

        for i in range(len(training)):
            # Note: training[i] is of the form (image, label)
            # Generate new data set containing (integral_image, label) pairs
            train_data.append( (integral_img(training[i][0]), training[i][1]) )

            # Initialize weights
            if training[i][1] == 1:
                weights[i] = 0.5 / n_pos
            else:
                weights[i] = 0.5 / n_neg

        features = self.build_features(train_data[0][0].shape)
        applied_features = self.apply_features(features, train_data)
        labels = np.array( [ data[1] for data in train_data ] )

        # TODO fix this optimization?
        '''
        indices = SelectPercentile(f_classif, percentile=10).fit(applied_features.T, labels).get_support(indices=True)
        print('TEST', type(features))
        applied_features = applied_features[indices]
        features = features[indices]
        '''

        for t in range(self.T):
            # Normalize weights
            weights = weights / np.linalg.norm(weights)

            # Train weak classifiers
            weak_classifiers = self.train_weak(applied_features, labels, features, weights)
            cl, error, accuracy = self.select_best(weak_classifiers, weights,
                    train_data)

            beta = error / (1.0 - error)

            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1-accuracy[i]))

            self.alphas.append(math.log( 1.0/beta ))
            self.classifiers.append(cl)

    # Classify an image with trained classifier
    def classify(self, image):

        total = 0
        ii = integral_img(image)

        for alpha, cl in zip(self.alphas, self.classifiers):
            total += alpha * cl.classify(ii)

        return 1 if total >= 0.5 * sum(self.alphas) else 0

    # Save this object to a file
    def save(self, filename):

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        print('Saved classifier to', filename)

    # Load saved object (WARNING: this is insecure, make sure file is trusted)
    @staticmethod
    def load(filename):

        with open(filename, 'rb') as f:
            return pickle.load(f)

