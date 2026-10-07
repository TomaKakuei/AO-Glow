"""Kaleid's predicted high-order error from noisy images and a nominal prior.

This is an estimator, not true WFE. Each independently deployed branch uses
only its own frozen network. It must never be called on assembled joint data.
"""
import numpy as np


class LearnedMerit:
    def __init__(self, model, branch, high_response):
        if branch not in ('front', 'rear') or model.branch != branch:
            raise ValueError('This merit needs one matching isolated branch')
        columns = slice(0, 15) if branch == 'front' else slice(15, 35)
        prior = np.asarray(high_response, dtype=np.float64)[:, columns]
        gram = prior.T@prior
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        self.factor = np.sqrt(np.maximum(eigenvalues, 0.))[:, None]*eigenvectors.T
        self.model = model
        self.branch = branch

    def vector(self, image):
        return self.factor@self.model.predict(self.branch, image)


def score_pair(vectors):
    a, b = vectors
    mean = (a+b)/2.
    noise_direction = (a-b)/np.sqrt(2.)
    # Empirical rank-one covariance from this pair. This is an estimated
    # acceptance uncertainty, not a calibrated frequentist coverage claim.
    variance = 2.*float(mean@noise_direction)**2 + float(noise_direction@noise_direction)**2
    return float(a@b), variance


def compare_vectors(before, after):
    b, bv = score_pair(before)
    a, av = score_pair(after)
    stderr = float(np.sqrt(max(bv+av, 0.)))
    improvement = b-a
    return dict(before=b, after=a, improvement=improvement, stderr=stderr,
                accepted=bool(improvement > 1.645*stderr))
