from latent_model_optimization import *
import numpy as np
from numpy import exp
from scipy.optimize import minimize, minimize_scalar
import math
import sympy as sym
from sympy.utilities.lambdify import lambdify

from pylab import *


class MingLeiModel(LatentModelOptimizer):

    def __init__(self, init_hidden_vars, init_params):
        super(MingLeiModel, self).__init__(init_hidden_vars, init_params)

        print("initializing...")

        # generate lambda function of jacobin of model with respect to hidden variables
        h1, h2 = sym.symbols('h1, h2', real=True)
        theta1, theta2 = sym.symbols('theta1, theta2', real=True)
        jac_hidden_sym_ = MingLeiModel._latent_model_jac_hidden_sym(h1, h2, theta1, theta2)
        self._jac_hidden = lambdify(
            (h1, h2, theta1, theta2),
            jac_hidden_sym_, modules='numpy')

        # generate lambda function of jacobin of model with respect to parameters
        h1, h2 = sym.symbols('h1, h2', real=True)
        theta1, theta2 = sym.symbols('theta1, theta2', real=True)
        jac_params_sym_ = MingLeiModel._latent_model_jac_params_sym(h1, h2, theta1, theta2)
        self._jac_params = lambdify(
            (h1, h2, theta1, theta2),
            jac_params_sym_, modules='numpy')

        # generate lambda function of Hessian of model with respect to parameters
        h1, h2 = sym.symbols('h1, h2', real=True)
        theta1, theta2 = sym.symbols('theta1, theta2', real=True)
        jac_params_sym_ = MingLeiModel._latent_model_hes_params_sym(h1, h2, theta1, theta2)
        self._hes_params = lambdify(
            (h1, h2, theta1, theta2),
            jac_params_sym_, modules='numpy')

        # test lambda function of jacobins
        #print(self._jac_params(0.1, 0.1, 0.1, 0.1))
        #print(self._jac_hidden(0.1, 0.1, 0.1, 0.1))
        #print(self._hes_params(0.1, 0.1, 0.1, 0.1))

        # generate lambda function of latent model
        h1, h2 = sym.symbols('h1, h2', real=True)
        theta1, theta2 = sym.symbols('theta1, theta2', real=True)
        self._latent_model = lambdify(
            (h1, h2, theta1, theta2),
            sym.re(MingLeiModel._latent_model_sym(h1, h2, theta1, theta2)), modules='numpy')

        print("initialization done!")


    @classmethod
    def _latent_model_sym(cls, h1, h2, theta1, theta2):
        """
        Symbolic latent model
        :param h1:
        :param h2:
        :param theta1:
        :param theta2:
        :return: symbolic model
        """
        beta = 2 * math.pi / 1.55e-9
        L = 100e-6
        k = math.sqrt(0.5)
        t = math.sqrt(0.5)

        M2 = sym.Matrix([[t, -k],
                         [k, t]])
        M4 = sym.Matrix([[t, -k],
                         [k, t]])

        e1 = sym.sqrt(h1) * sym.exp(-1j * h2)
        e2 = sym.sqrt(1 - h1)
        Ein = sym.Matrix([[e1], [e2]])
        M10 = sym.Matrix([[sym.exp(-1j * beta * L - 1j * theta1), 0], [0, sym.exp(-1j * beta * L)]])
        M30 = sym.Matrix([[sym.exp(-1j * beta * L - 1j * theta2), 0], [0, sym.exp(-1j * beta * L)]])

        M50 = M4 * M30 * M2 * M10 * Ein
        r30 = M50[0, 0]

        p3 = sym.conjugate(r30) * r30

        return p3

    @classmethod
    def _latent_model_jac_params_sym(cls, h1, h2, theta1, theta2):
        """
        Symbolic model Jacobin with respect to model parameters
        :param h1:
        :param h2:
        :param theta1:
        :param theta2:
        :return:
        """

        p3 = MingLeiModel._latent_model_sym(h1, h2, theta1, theta2)

        dydh1 = sym.re(sym.diff(p3, theta1))
        dydh2 = sym.re(sym.diff(p3, theta2))

        return sym.Matrix([dydh1, dydh2]).transpose()

    @classmethod
    def _latent_model_hes_params_sym(cls, h1, h2, theta1, theta2):
        """
        Symbolic model Jacobin with respect to model parameters
        :param h1:
        :param h2:
        :param theta1:
        :param theta2:
        :return:
        """

        p3 = MingLeiModel._latent_model_sym(h1, h2, theta1, theta2)

        dydh1 = sym.diff(p3, theta1)
        dydh2 = sym.diff(p3, theta2)

        dydh1h1 = sym.re(sym.diff(dydh1, theta1))
        dydh1h2 = sym.re(sym.diff(dydh1, theta2))
        dydh2h1 = sym.re(sym.diff(dydh2, theta1))
        dydh2h2 = sym.re(sym.diff(dydh2, theta2))

        return sym.Matrix([[dydh1h1, dydh1h2], [dydh2h1, dydh2h2]])

    @classmethod
    def _latent_model_hes_hidden_sym(cls, h1, h2, theta1, theta2):
        """
        Symbolic model jacobin with respect to hidden variables
        :param h1:
        :param h2:
        :param theta1:
        :param theta2:
        :return:
        """

        p3 = MingLeiModel._latent_model_sym(h1, h2, theta1, theta2)

        dydh1 = sym.re(sym.diff(p3, h1))
        dydh2 = sym.re(sym.diff(p3, h2))

        dydh1h1 = sym.re(sym.diff(dydh1, h1))
        dydh1h2 = sym.re(sym.diff(dydh1, h2))
        dydh2h1 = sym.re(sym.diff(dydh2, h1))
        dydh2h2 = sym.re(sym.diff(dydh2, h2))

        return sym.Matrix([[dydh1h1, dydh1h2], [dydh2h1, dydh2h2]])

    @classmethod
    def _latent_model_jac_hidden_sym(cls, h1, h2, theta1, theta2):
        """
        Symbolic model jacobin with respect to hidden variables
        :param h1:
        :param h2:
        :param theta1:
        :param theta2:
        :return:
        """

        p3 = MingLeiModel._latent_model_sym(h1, h2, theta1, theta2)

        dydh1 = sym.re(sym.diff(p3, h1))
        dydh2 = sym.re(sym.diff(p3, h2))

        return sym.Matrix([dydh1, dydh2]).transpose()

    def latent_model(self, hidden_vars, params):
        # Minglei upadte: h1: amplitude of e1 vector; h2: relative phase between e1 and e2
        h1, h2 = hidden_vars
        theta1, theta2 = params

        return self._latent_model(h1, h2, theta1, theta2)

    def model_jac(self, params):
        h1, h2 = self.hidden_vars
        theta1, theta2 = params

        h1 = min(h1, 1.0)

        jac_tmp = self._jac_params(h1, h2, theta1, theta2)
        return np.ndarray(shape=[2], buffer=np.array([jac_tmp[0][0], jac_tmp[0][1]]))

    def functional_jac(self, hidden_vars):
        h1, h2 = hidden_vars
        theta1, theta2 = self.params

        h1 = min(h1, 1.0)
        return self._jac_hidden(h1, h2, theta1, theta2)

    def validate_model(self):
        """
        if 0 <= self.params[0] <= 2 * math.pi and \
                                0 <= self.params[1] <= 2 * math.pi and \
                                0 <= self.hidden_vars[0] <= 1:
            return True
        else:
            raise ValueError
        """
        pass

    def model_hes(self, params):
        h1, h2 = self.hidden_vars
        theta1, theta2 = params

        h1 = min(h1, 1.0)
        return self._hes_params(h1, h2, theta1, theta2)

    def train_and_optimize(self, sample_numbers=[4, 4], optimize_method='Newton-CG', jac=True):

        # train model
        sample_params1 = np.linspace(0, math.pi, sample_numbers[0])
        sample_params2 = np.linspace(0, math.pi, sample_numbers[1])
        sample_params = []
        for sample1 in sample_params1:
            for sample2 in sample_params2:
                sample_params.append([sample1, sample2])

        self.train(sample_parameters=sample_params, bounds=([0, 0], [1, 2 * math.pi]), method='trf', jac=True)

        self.validate_model()

        init_guess = [
            [np.pi, np.pi],
            [0.1*np.pi, np.pi * 0.1],
            [0.1*np.pi, np.pi * 1.8],
            [1.8*np.pi, np.pi * 1.8],
            [1.8*np.pi, np.pi * 0.1]
        ]

        # minimize the estimated model
        def map2domain(theta):
            while theta > 2 * np.pi:
                theta -= 2 * np.pi
            while theta < 0:
                theta += 2 * np.pi
            return theta

        for ini in init_guess:
            print(ini)
            if not jac:
                results = minimize(self.model,
                                   np.ndarray(shape=[2], buffer=np.array(ini)),
                                   method=optimize_method, jac=False, tol=1E-16
                                   )
            else:
                results = minimize(self.model,
                                   np.ndarray(shape=[2], buffer=np.array(ini)),
                                   method=optimize_method, jac=self.model_jac, hess=self.model_hes, tol=1E-16
                                   )

            if results.success and results.fun < 1E-5:
                print("minimum is " + str(results.fun))
                ps = list(map(
                    map2domain, results.x
                ))

                if ps[0] > math.pi:
                    ps[0] -= math.pi
                    ps[1] = 2*math.pi-ps[1]

                self.set_params(ps)
                break

        self.validate_model()
