
from math import exp, log, sqrt

# implementation taken from kaggle scripts:
# https://www.kaggle.com/sudalairajkumar/outbrain-click-prediction/ftrl-starter-with-leakage-vars/code


def hash_element(el, D):
    h = hash(el) % D
    if h < 0:
        h = h + D
    return h

def hash_elements(elements, D):
    return [hash_element(el, D) for el in elements]


class FtrlProximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interactions):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D

        self.interactions = interactions

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.0] * (D + 1)
        self.z = [0.0] * (D + 1)
        self.w = {}

    def to_indices(self, x):
        res = hash_elements(x, self.D)

        if self.interactions:
            sorted_x = sorted(x)
            len_x = len(sorted_x)

            for i in range(len_x):
                for j in range(i + 1, len_x):
                    h = hash_element(sorted_x[i] + '_' + sorted_x[j], self.D)
                    res.append(h)

        return res

    def predict(self, x):
        x_hashed = self.to_indices(x)
        return self.predict_hashed(x_hashed)

    def predict_hashed(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.

        indices = [0]
        for i in x:
            indices.append(i + 1)

        for i in indices:
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.0
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1.0 / (1.0 + exp(-max(min(wTx, 35.0), -35.0)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: a list of indices
                p: probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        indices = [0]
        for i in x:
            indices.append(i + 1)

        # update z and n
        for i in indices:
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g

    def fit(self, x, y):
        x_hashed = self.to_indices(x)
        p = self.predict_hashed(x_hashed)
        self.update(x_hashed, p, y)