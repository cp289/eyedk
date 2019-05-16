#!/usr/bin/python3

import math
import numpy as np
import cv2 as cv
import pickle

# Compute the integral image of an input image
def integral_img(img):
    ii = np.zeros(img.shape)
    s = np.zeros(img.shape)

    for y in range(len(img)):
        for x in range(len(img[y])):
            s[y][x] = s[y-1][x] + img[y][x] if y >= 1 else img[y][x]
            ii[y][x] = ii[y][x-1] + s[y][x] if x >= 1 else s[y][x]

    return ii

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    # Compute region sum using integral image
    def compute_feature(self, ii):
        return ii[self.y+self.h][self.x+self.w] - ii[self.y+self.h][self.x] - ii[self.y][self.x+self.w] + ii[self.y][self.x]

# Weak classifier
class WeakClass:

    def __init__(self, regs_pos, regs_neg, threshold, polarity):
        self.regs_pos = regs_pos
        self.regs_neg = regs_neg
        self.threshold = threshold
        self.polarity = polarity

    # Viola-Jones weak classifier function
    def classify(self, int_img):
        feature = lambda ii: sum([p.compute_feature(ii) for p in
            self.regs_pos]) - sum([n.compute_feature(ii) for n in
                self.regs_neg])
        return 1 if self.polarity*feature(int_img) < self.polarity*self.threshold else 0


# Class for the Viola Jones algorithm
class ViolaJones:

    def __init__(self, T = 10):
        self.T = T
        self.alphas = []
        self.classifiers = []

    # Build list of ordered pairs of features ( [pos], [neg] ). The values of
    # interest are of the form (sum(pos) - sum(neg))
    def build_features(self, img_shape):

        print('Building features...')

        H, W = img_shape
        features = []

        for w in range(1, W+1):
            for h in range(1, H+1):
                i = 0
                while i + w < W:
                    j = 0
                    while j + h < H:
                        # Generate regions
                        current = Rect(i, j, w, h)
                        right = Rect(i+w, j, w, h)
                        bottom = Rect(i, j+h, w, h)
                        right_2 = Rect(i+2*w, j, w, h)
                        bottom_2 = Rect(i, j+2*h, w, h)
                        bottom_right = Rect(i+w, j+h, w, h)

                        # Add features if regions are within image
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

    def apply_features(self, features, train_data):

        print('Applying features...')

        x = np.zeros( (len(features), len(train_data)) )
        y = np.array( list(map(lambda data: data[1], train_data)) )

        i = 0
        for reg_pos, reg_neg in features:
            feature = lambda ii: sum([p.compute_feature(ii) for p in reg_pos]) - sum([n.compute_feature(ii) for n in reg_neg])
            x[i] = list(map(lambda data: feature(data[0]), train_data))
            i += 1

        return x, y

    # Train weak classifiers
    def train_weak(self, x, y, features, weights):

        ptotal = ntotal = 0

        for w, label in zip(weights, y):
            if label == 1:
                ptotal += w
            else:
                ntotal += w

        classifiers = []
        nfeatures = x.shape[0]

        for index, feature in enumerate(x):

            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers),
                    nfeatures), end='\r' )

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
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

        print("Trained %d classifiers out of %d" % (len(classifiers), nfeatures) )

        return classifiers

    # Find best weak classifier by calculating average weighted error
    def select_best(self, classifiers, weights, train_data):

        best_classifier = None
        best_error = math.inf
        best_accuracy = None

        for cl in classifiers:
            error = 0 # TODO does this actually make sense??
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

        weights = np.zeros(len(training))
        train_data = []

        for x in range(len(training)):
            train_data.append( (integral_img(training[x][0]), training[x][1]) )

            # Initialize weights
            if training[x][1] == 1:
                weights[x] = 0.5 / n_pos
            else:
                weights[x] = 0.5 / n_neg

        features = self.build_features(train_data[0][0].shape)
        x, y = self.apply_features(features, train_data)

        for t in range(self.T):
            # Normalize weights
            weights = weights / np.linalg.norm(weights)

            # Train weak classifiers
            weak_classifiers = self.train_weak(x, y, features, weights)
            cl, error, accuracy = self.select_best(weak_classifiers, weights,
                    train_data)

            beta = error / (1.0 - error)

            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1-accuracy[i]))

            self.alphas.append(math.log( 1.0/beta ))
            self.classifiers.append(cl)

    # Classify an image
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

    # Load saved object (WARNING: this is insecure, make sure file is trusted)
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    print('L')

