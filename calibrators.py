import logging as log
import math
import random
import collections

import numpy as np
from scipy.optimize import least_squares, newton_krylov, broyden1, approx_fprime, check_grad, newton, fsolve


class SlidingWindowCalibrator(object):
    def __init__(self, model, num_samples=6, num_resamples=1):
        self.model = model
        self.num_samples = num_samples
        self.num_resamples = num_resamples

        self.model_params = None
        self.sample_queue = None

    def initialize(self, seed=3, num_observes=1):

        log.info("sliding window calibrator initializing...")

        log.info("sweeping..")
        sample_inputs = list()
        random.seed(seed)
        for _ in range(self.num_samples):
            sample_inputs.append([
                random.uniform(0, math.pi),
                random.uniform(0, math.pi)
            ])

        log.debug("sweeping model inputs: " + str(sample_inputs))
        observes_ = [[self.model.observe(input_) for _ in range(num_observes)] for input_ in sample_inputs]
        observes_ = np.mean(np.asarray(observes_), axis=1).tolist()
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

    def track(self, epsilon=None, threshold=1E-6, num_observes=1, verbose=0, tol=1E-6):
        if self.sample_queue is None or self.model_params is None:
            raise ValueError("calibrator not initialized!")

        if not isinstance(num_observes, int) or num_observes < 1:
            raise ValueError("number of observes must be a integer greater than or equal to 1!")

        # estimate output
        new_output = 0
        for _ in range(num_observes):
            new_output += self.model.observe(self.sample_queue[-1].inputs)
        new_output /= num_observes
        if np.abs(new_output - self.model.guess(self.sample_queue[-1].inputs, self.model_params)) < threshold:
            return

        epsilon = np.sqrt(np.finfo(float).eps) if epsilon is None else epsilon
        log.debug("epsilon: " + str(epsilon))

        directions = [
            [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]
        ]
        success = False

        for direct in directions:

            log.debug("direction: " + str(direct))
            # estimate f prime times epsilon
            numerical_output_changes = None
            for _ in range(num_observes):
                if numerical_output_changes is None:
                    numerical_output_changes = self._approx_fprime_epsilon(
                        self.sample_queue[-1].inputs, self.model.observe, [direct[0] * epsilon, direct[1] * epsilon]
                    )
                else:
                    numerical_output_changes += self._approx_fprime_epsilon(
                        self.sample_queue[-1].inputs, self.model.observe, [direct[0] * epsilon, direct[1] * epsilon]
                    )
            numerical_output_changes /= num_observes

            def residual(new_params):
                log.debug("tried parameters: " + str(new_params))
                jac_inputs = np.squeeze(self.model.guess_jac_inputs(
                    self.sample_queue[-1].inputs, new_params
                ))

                jac_residual = numerical_output_changes - jac_inputs * epsilon

                return jac_residual

            def residual_jac(new_params):

                hes_params = self.model.guess_hes_inputs_params(
                    self.sample_queue[-1].inputs, new_params
                )

                return - hes_params * epsilon

            if verbose > 0:
                print('tracking...')
            results = fsolve(residual, np.asarray(self.model_params), fprime=residual_jac, xtol=3e-16)

            next_model_params = results.tolist()
            log.debug(results)

            if verbose > 0:
                print('searching minimum...')
            min_inputs = self.model.argmin(params=next_model_params, initial_inputs=self.sample_queue[-1].inputs,
                                           verbose=verbose)
            min_ = self.model.observe(min_inputs)

            log.info("minimum of proposed model: " + str(min_))

            if min_ < new_output + tol:
                self.model_params = next_model_params
                self.sample_queue.popleft()
                self.sample_queue.append(SampleRecord(min_inputs, min_))
                success = True
                break

        if not success:
            raise ValueError("Failed to find better minimum!")

    @staticmethod
    def _approx_fprime_epsilon(xk, f, epsilon, args=(), f0=None):
        if f0 is None:
            f0 = f(*((xk,) + args))
        grad = np.zeros((len(xk),), float)
        ei = np.zeros((len(xk),), float)
        for k in range(len(xk)):
            ei[k] = 1.0
            d = epsilon * ei
            if epsilon[k] >= 0:
                grad[k] = f(*((xk + d,) + args)) - f0
            else:
                grad[k] = f0 - f(*((xk + d,) + args))
            ei[k] = 0.0
        return grad


class PointLSCalibrator(object):
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

    def track(self, epsilon=None, threshold=1E-6, num_observes=1, verbose=0):
        if self.sample_queue is None or self.model_params is None:
            raise ValueError("calibrator not initialized!")

        if not isinstance(num_observes, int) or num_observes < 1:
            raise ValueError("number of observes must be a integer greater than or equal to 1!")

        # estimate output
        new_output = 0
        for _ in range(num_observes):
            new_output += self.model.observe(self.sample_queue[-1].inputs)
        new_output /= num_observes
        if np.abs(new_output - self.model.guess(self.sample_queue[-1].inputs, self.model_params)) < threshold:
            return

        epsilon = np.sqrt(np.finfo(float).eps) if epsilon is None else epsilon
        log.debug("epsilon: " + str(epsilon))

        # observe other points
        y1 = 0
        x1 = np.asarray(self.sample_queue[-1].inputs) + np.asarray([0, epsilon])
        for _ in range(num_observes):
            y1 += self.model.observe(x1.tolist())
        y1 /= num_observes

        y2 = 0
        x2 = np.asarray(self.sample_queue[-1].inputs) + np.asarray([epsilon, 0])
        for _ in range(num_observes):
            y2 += self.model.observe(x2.tolist())
        y2 /= num_observes

        def residual(new_params):

            y0_model = self.model.guess(self.sample_queue[-1].inputs, new_params)
            y1_model = self.model.guess(x1.tolist(), new_params)
            y2_model = self.model.guess(x2.tolist(), new_params)

            return np.asarray([new_output, y1, y2]) - np.asarray([y0_model, y1_model, y2_model])

        def residual_jac(new_params):

            jac_y0 = self.model.guess_jac_params(self.sample_queue[-1].inputs, new_params)
            jac_y1 = self.model.guess_jac_params(x1.tolist(), new_params)

            return np.concatenate(jac_y0, jac_y1)

        if verbose > 0:
            print('tracking...')
        results = least_squares(residual, self.model_params, ftol=1e-16, xtol=1e-16, gtol=1e-16)

        next_model_params = results.x.tolist()
        log.debug(results)

        if verbose > 0:
            print('searching minimum...')
        min_inputs = self.model.argmin(params=next_model_params, initial_inputs=self.sample_queue[-1].inputs, verbose=verbose)
        min_ = self.model.observe(min_inputs)

        log.info("minimum of proposed model: " + str(min_))

        if min_ < new_output:
            self.model_params = next_model_params
            self.sample_queue.popleft()
            self.sample_queue.append(SampleRecord(min_inputs, min_))

    @staticmethod
    def _approx_fprime_epsilon(xk, f, epsilon, args=(), f0=None):
        if f0 is None:
            f0 = f(*((xk,) + args))
        grad = np.zeros((len(xk),), float)
        ei = np.zeros((len(xk),), float)
        for k in range(len(xk)):
            ei[k] = 1.0
            d = epsilon * ei
            grad[k] = f(*((xk + d,) + args)) - f0
            ei[k] = 0.0
        return grad


class SampleRecord(object):
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output
        self.decay_ = 1.0

    def decay(self, decay):
        self.decay_ *= decay