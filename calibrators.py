import logging as log
import math
import random
import collections

import numpy as np
from scipy.optimize import least_squares, newton_krylov, broyden1, approx_fprime, check_grad, newton, fsolve



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

    def track(self, verbose=0):
        if self.sample_queue is None or self.model_params is None:
            raise ValueError("calibrator not initialized!")

        new_output = self.model.observe(self.sample_queue[-1].inputs)

        epsilon = np.sqrt(np.finfo(float).eps)
        log.debug("epsilon: " + str(epsilon))
        numerical_output_changes = self._approx_fprime_epsilon(
            self.sample_queue[-1].inputs, self.model.observe, [epsilon, epsilon]
        )

        def residual(new_params):
            log.debug("tried parameters: " + str(new_params))
            jac_inputs = np.squeeze(self.model.guess_jac_inputs(
                self.sample_queue[-1].inputs, new_params
            ))

            output_residual = new_output - self.model.guess(self.sample_queue[-1].inputs, new_params)
            jac_residual = numerical_output_changes - jac_inputs * epsilon

            # return np.asarray(jac_residual.tolist() + [output_residual,])
            return jac_residual

        def residual_jac(new_params):

            jac_params = self.model.guess_jac_params(
                self.sample_queue[-1].inputs, new_params
            )

            hes_params = self.model.guess_hes_inputs_params(
                self.sample_queue[-1].inputs, new_params
            )

            return -np.concatenate((hes_params * epsilon, jac_params), axis=0) * 1E8

        # TODO: remove
        print('target residual: ' + str(residual(self.model.params).tolist()))

        # test_params = [0.5, 0.5]
        # print('target residual jac:' + str(residual_jac(test_params).tolist()))
        # print('check residual jac: ' + str(
        #     check_grad(lambda params_: residual(params_)[0], lambda params_: residual_jac(params_)[0, :],
        #                test_params, epsilon=epsilon)))
        # print('check residual jac: ' + str(
        #     check_grad(lambda params_: residual(params_)[1], lambda params_: residual_jac(params_)[1, :],
        #                test_params, epsilon=epsilon)))
        # print('check residual jac: ' + str(
        #     check_grad(lambda params_: residual(params_)[2], lambda params_: residual_jac(params_)[2, :],
        #                test_params, epsilon=epsilon)))

        if verbose > 0:
            print('tracking...')
        # results = least_squares(
        #     residual, self.model_params, verbose=verbose, method='trf', bounds=self.model.params_bounds,
        #     ftol=3e-16, xtol=3e-16, gtol=3e-16
        # )
        results = fsolve(residual, np.asarray(self.model_params), xtol=3e-16)

        next_model_params = results.tolist()
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

    # def track(self, verbose=0):
    #     if self.sample_queue is None or self.model_params is None:
    #         raise ValueError("calibrator not initialized!")
    #
    #     new_output = self.model.observe(self.sample_queue[-1].inputs)
    #
    #     def change_residule(new_params):
    #         jac_params = np.squeeze(self.model.guess_jac_params(
    #             self.sample_queue[-1].inputs, new_params
    #         ))
    #
    #         change = (np.asarray(new_params) - np.asarray(self.model_params)).tolist()
    #
    #         residual_ = np.asarray([new_output, new_output]) - \
    #                     np.asarray([
    #                         jac_params[0] * change[0] / 2.0,
    #                         jac_params[1] * change[1] / 2.0
    #                     ])
    #
    #         print("new parameters " + str(new_params))
    #         print("residule" + str(residual_))
    #         return residual_
    #
    #     def change_residule_jac(new_params):
    #         jac_params = np.squeeze(self.model.guess_jac_params(
    #             self.sample_queue[-1].inputs, new_params
    #         ))
    #
    #         hes_params = self.model.guess_hes_params(
    #             self.sample_queue[-1].inputs, new_params
    #         )
    #
    #         change = np.asarray(new_params) - np.asarray(self.model_params)
    #
    #         h00 = - change[0] * hes_params[0, 0] - jac_params[0]
    #         h01 = - change[0] * hes_params[0, 1]
    #         h10 = - change[1] * hes_params[1, 0]
    #         h11 = - change[1] * hes_params[1, 1] - jac_params[1]
    #
    #         return np.asarray([[h00, h01], [h10, h11]]) / 2.0
    #
    #     # TODO: remove
    #     print('target residual: ' + str(change_residule(self.model.params).tolist()))
    #
    #     if verbose > 0:
    #         print('tracking...')
    #     results = least_squares(
    #         change_residule, self.model_params, verbose=verbose, method='trf', bounds=self.model.params_bounds,
    #         jac=change_residule_jac,
    #         ftol=3e-16, xtol=3e-16, gtol=3e-16
    #     )
    #
    #     next_model_params = results.x.tolist()
    #     log.debug(results)
    #
    #     if verbose > 0:
    #         print('searching minimum...')
    #     min_inputs = self.model.argmin(params=next_model_params, initial_inputs=self.sample_queue[-1].inputs, verbose=verbose)
    #     min_ = self.model.observe(min_inputs)
    #
    #     log.info("minimum of proposed model: " + str(min_))
    #
    #     if min_ < new_output:
    #         self.model_params = next_model_params
    #         self.sample_queue.popleft()
    #         self.sample_queue.append(SampleRecord(min_inputs, min_))


class SampleRecord(object):
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output
        self.decay_ = 1.0

    def decay(self, decay):
        self.decay_ *= decay