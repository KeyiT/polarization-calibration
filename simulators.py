from minglei_complete_model import MingLeiModel
import hp816x_instr
import time
import Ke26XXA
from Ke26XXA import Ke26XXA
import math
import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np


# from sympy.slovers import solve
# from sympy import Symbol

class PhysicalModel(object):
    def __init__(self, opt_slot, opt_obv_chn, keith_dev, keith_chn1, keith_chn2,
                 keith_imax):
        self.opt_slot = opt_slot
        self.opt_obv_chn = opt_obv_chn
        self.keith_dev = keith_dev
        self.keith_chn1 = keith_chn1
        self.keith_chn2 = keith_chn2
        self.keith_imax = keith_imax
        self.hp_mainframe = hp816x_instr.hp816x()

    #        print('ok')

    def observe(self, inputs):
        # TODO: return p3 from your machine (in power unit)
        self.__set_inputs(inputs)
        #        time.sleep(0.001)  ##unit: s
        p3 = self.hp_mainframe.readPWM(self.opt_slot, self.opt_obv_chn)
        p3_normalized = p3 / 39.81e-06
        p3_dBm = 10 * np.log10(p3) + 30
        #        print(p3_dBm)
        return p3_normalized

    def Output_PWM(self, opt_chn):
        #        time.sleep(0.001)  ##unit: s
        p4 = self.hp_mainframe.readPWM(self.opt_slot, opt_chn)
        p4_dBm = 10 * np.log10(p4) + 30
        #        print(p4_dBm)
        return p4_dBm

    def hpmainframe_laser(self, state):
        self.hp_mainframe.setTLSState(state, slot='auto')

    def hpmainframe_connect(self, visaAddr):
        self.hp_mainframe.connect(visaAddr, reset=0, forceTrans=1)  # connects to the laser

    def setAvgtime(self, Avgtime):
        self.hp_mainframe.setPWMAveragingTime(1, 1, Avgtime)  # unit: s
        self.hp_mainframe.setPWMAveragingTime(1, 0, Avgtime)

    def setPWMUnit(self, Unit):
        self.hp_mainframe.setPWMPowerUnit(self.opt_slot, 0, Unit)
        self.hp_mainframe.setPWMPowerUnit(self.opt_slot, 1, Unit)

    def on(self):
        self.keith_dev.outputEnable(self.keith_chn1, True)
        self.keith_dev.outputEnable(self.keith_chn2, True)

    def off(self):
        self.keith_dev.outputEnable(self.keith_chn1, False)
        self.keith_dev.outputEnable(self.keith_chn2, False)

    def getCurrent(self):
        currents = [self.keith_dev.getCurrent(self.keith_chn1), self.keith_dev.getCurrent(self.keith_chn2)]
        return currents

    def __set_inputs(self, inputs):
        alpha = 104.2
        beta = -0.07855
        #        R = 200
        print(inputs)
        #        a = []
        #        b = []
        #        eqn1 = a[0]*current1**4 + a[1]*current1**3 + a[2]*current1**2 - (params[0] - beta)/alpha
        #        eqn2 = b[0]*current2**4 + b[1]*current1**3 + b[2]*current1**2 - (params[1] - beta)/alpha
        #        slove(eqn1, current1, )

        inter_ = [
            inputs[0] - beta,
            inputs[1] - beta
        ]
        inter_ = list(map(
            lambda xx: xx + 2 * np.pi if xx < 0 else xx,
            inter_
        ))

        coeff = [
            [76880.4089, 49422.4911, 42.442, 196.94, 0, -inter_[0] / alpha],
            [76880.4089, 49422.4911, 42.442, 196.94, 0, -inter_[1] / alpha]
        ]
        roots = [np.roots(coeff[0]), np.roots(coeff[1])]

        for i in range(0, len(roots)):
            roots_ = roots[i]
            roots_ = list(filter(
                lambda r_: np.abs(np.imag(r_)) < 1E-8,
                roots_
            ))
            roots[i] = list(filter(
                lambda r_: True if (r_ >= 0) and (r_ < 0.036) else False,
                roots_
            ))

            if len(roots[i]) == 0:
                raise ValueError("invalid roots")

        #        print("\ncurrent: ")
        #        print(roots)

        """
        # TODO: set theta1 and theta2 to your machine. params=[theta1, theta2]
        if params[0] < beta and params[1] < beta:
            current1 = math.sqrt((params[0]-beta+2*math.pi)/(alpha*R))
            current2 = math.sqrt((params[1]-beta+2*math.pi)/(alpha*R))
        if params[0] < beta and params[1] > beta:
            current1 = math.sqrt((params[0]-beta+2*math.pi)/(alpha*R))
            current2 = math.sqrt((params[1]-beta)/(alpha*R))
        if params[0] > beta and params[1] < beta:
            current1 = math.sqrt((params[0]-beta)/(alpha*R))
            current2 = math.sqrt((params[1]-beta+2*math.pi)/(alpha*R))
        if params[0] > beta and params[1] > beta:
            current1 = math.sqrt((params[0]-beta)/(alpha*R))
            current2 = math.sqrt((params[1]-beta)/(alpha*R))         
        if current1 > self.keith_imax + 1e-3:
            print('current larger than maximum channel1 current')
            current1 = math.sqrt((params[0]-beta-2*math.pi)/(alpha*R))
        if current2 > self.keith_imax + 1e-3:
            print('current larger than maximum channel2 current')
            current2 = math.sqrt((params[1]-beta-2*math.pi)/(alpha*R))
        """
        self.keith_dev.setCurrent(self.keith_chn1, roots[0][0])
        self.keith_dev.setCurrent(self.keith_chn2, roots[1][0])
        ##need a response time?


class SimulatorModel(object):

    def __init__(self, params):

        print("initializing...")
        # generate lambda function of latent model
        h1, h2 = sym.symbols('h1, h2', real=True)
        theta1, theta2 = sym.symbols('theta1, theta2', real=True)
        self._model = lambdify(
            (h1, h2, theta1, theta2),
            sym.re(self._model_sym(h1, h2, theta1, theta2)),
            modules='numpy'
        )

        self.params = params

        print("initialization done!")

    @classmethod
    def _model_sym(cls, h1, h2, theta1, theta2):
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

    def observe(self, inputs):
        h1, h2 = self.params
        theta1, theta2 = inputs

        return self._model(h1, h2, theta1, theta2)


# instruction:
# 1. implement observe and set_params in PhysicalModel
# 2. initialize: model = PhysicalModel([0.5, 0.5], [0.5, 0.5])
# 3. set slot and channel: model.set_slot_chn(1, 1)
# 4. find h1, h2, theta1, theta2: model.train_and_optimize()
# 5. get it run along with your machine. you can keep calling
# model.train_and_optimize() to calibrate your machine in each time step