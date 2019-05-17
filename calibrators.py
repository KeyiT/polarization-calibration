import logging as log
import math
import random
import collections

import numpy as np
from scipy.optimize import least_squares, newton_krylov, broyden1



class SlidingWindownCalibrator(object):
    def __init__(self, model, num_samples=6, num_resamples=1):
        self.model = model
        self.num_samples = num_samples
        self.num_resamples = num_resamples

        self.model_params = None
        self.sample_queue = None

    def initialize(self, seed=3):

        log.info("sliding widown calibrator initializing...")

        log.info("sweeping..")
        sample_inputs = list()
        random.seed(seed)
        for _ in range(self.num_samples):
            sample_inputs.append([
                random.uniform(0, math.pi),
                random.uniform(0, math.pi)
            ])

        log.debug("sweeping model inputs: " + str(sample_inputs))
        observes_ = [self.model.observe(input_) for input_ in sample_inputs]
        log.debug("sweeping model outputs: " + str(observes_))

        self.sample_queue = collections.deque([
            SampleRecord(sample_inputs[i], observes_[i]) for i in range(self.num_samples)
        ])

        def loss_func_(model_params):
            loss = []
            for i in range(self.num_samples):
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

        log.info("searching initial model parameters...")
        for ini in init_guess:
            results = least_squares(
                loss_func_, ini, verbose=1, method='trf', bounds=self.model.params_bounds, ftol=3e-16, xtol=1e-16, gtol=3e-16
            )
            if results.success:
                self.model_params = results.x
                log.info("initial model parameter: " + str(self.model_params))

                min_inputs = self.model.argmin(params=self.model_params)
                min_ = self.model.observe(min_inputs)

                self.sample_queue.popleft()
                self.sample_queue.append(SampleRecord(min_inputs, min_))
                return

        log.info("model parameters not found!")

    def calibrate(self, seed=3, history_decay=0.9):
        if self.sample_queue is None or self.model_params is None:
            raise ValueError("calibrator not initialized!")

        random.seed(seed)
        for record_ in self.sample_queue:
            record_.decay(history_decay)

        for _ in range(self.num_resamples):
            # remove samples expired
            self.sample_queue.popleft()

            # resample
            inputs_ = [
                random.uniform(0, math.pi),
                random.uniform(0, math.pi)
            ]
            output_ = self.model.observe(inputs_)
            self.sample_queue.append(SampleRecord(inputs_, output_))

        def loss_func_(model_params):
            loss = []
            for i in range(self.num_samples):
                pred = self.model.guess(self.sample_queue[i].inputs, model_params)
                loss.append(self.model.residual(pred, self.sample_queue[i].output) * self.sample_queue[i].decay_)
            return loss

        results = least_squares(
            loss_func_, self.model_params, verbose=1, method='trf', bounds=self.model.params_bounds, ftol=3e-16, xtol=3e-18,
            gtol=3e-16
        )
        if results.success and np.sum(np.array(results.fun)) < 1E-7:
            self.model_params = results.x
            log.info("model parameter: " + str(self.model_params))
            return self.model_params

        log.info("model parameters not found!")
        if results.success:
            return results.x

    def track(self):
        if self.sample_queue is None or self.model_params is None:
            raise ValueError("calibrator not initialized!")

        new_output = self.model.observe(self.sample_queue[-1].inputs)

        def change_residule(change):
            jac_params = np.squeeze(self.model.guess_jac_params(
                self.sample_queue[-1].inputs, (np.asarray(change) + np.asarray(self.model_params)).tolist()
            ))

            residual_ = np.asarray([new_output, new_output]) - np.asarray([jac_params[0] * change[0], jac_params[1] * change[1]])
            return residual_ * 1E15

        #best_change = broyden1(change_residule, [0, 0], f_tol=1e-6, verbose=1)
        params_bounds = [
            [self.model.params_bounds[0][0]-self.model_params[0], self.model.params_bounds[0][1]-self.model_params[1]],
            [self.model.params_bounds[1][0] - self.model_params[0], self.model.params_bounds[1][1] - self.model_params[1]]
        ]
        print('tracking...')
        results = least_squares(
            change_residule, [1E-3, 1E-3], verbose=1, method='trf', bounds=params_bounds, ftol=3e-16,
            xtol=3e-16,
            gtol=3e-16
        )

        self.model_params = (results.x + np.asarray(self.model_params)).tolist()
        log.debug(results)

        print('searching minimum...')
        min_inputs = self.model.argmin(params=self.model_params, initial_inputs=self.sample_queue[-1].inputs)
        min_ = self.model.observe(min_inputs)

        self.sample_queue.popleft()
        self.sample_queue.append(SampleRecord(min_inputs, min_))


class SampleRecord(object):
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output
        self.decay_ = 1.0

    def decay(self, decay):
        self.decay_ *= decay