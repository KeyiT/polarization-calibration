import logging as log
import math
import random

import numpy as np
from scipy.optimize import least_squares


class SlidingWindownCalibrator(object):
    def __init__(self, model):
        self.model = model

        self.model_params = None

    def initialize(self, num_samples=6, seed=3):

        log.info("sliding widown calibrator initializing...")

        log.info("sweeping..")
        sample_inputs = list()
        random.seed(seed)
        for _ in range(num_samples):
            sample_inputs.append([
                random.uniform(0, math.pi),
                random.uniform(0, math.pi)
            ])

        log.debug("sweeping model inputs: " + str(sample_inputs))
        observes_ = [self.model.observe(input_) for input_ in sample_inputs]
        log.debug("sweeping model outputs: " + str(observes_))

        def loss_func_(model_params):
            loss = []
            for i in range(num_samples):
                pred = self.model.guess(sample_inputs[i], model_params)
                loss.append(self.model.residual(pred, observes_[i]))
            return loss

        init_guess = [
            [0.5, np.pi],
            [0.1, np.pi * 0.1],
            [0.1, np.pi * 1.8],
            [0.9, np.pi * 1.8],
            [0.9, np.pi * 0.1]
        ]

        for ini in init_guess:
            results = least_squares(
                loss_func_, ini, verbose=1, method='trf', bounds=self.model.params_bounds, ftol=3e-16, xtol=3e-16, gtol=3e-16
            )
            if results.success and np.sum(np.array(results.fun)) < 1E-7:
                self.model_params = results.x
                break