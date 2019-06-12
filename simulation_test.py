from minglei_complete_model import MingLeiModel
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import math
from pylab import *
import json


class TestModel(MingLeiModel):

    def __init__(self, init_hidden_vars, init_params, target_hidden_vars):
        self.target_hidden_vars = target_hidden_vars
        super(TestModel, self).__init__(init_hidden_vars, init_params)

    def observe(self, params):
        return self.latent_model(
            self.target_hidden_vars,
            params)



target_hidden_vars = [0.85991353, 0.2477762]
model = TestModel([0.5, 0.5], [0.5, 0.5])

model.train_and_optimize([5, 5])
print("\nhidden variables:")
print(model.hidden_vars)
print("target hidden variables:")
print(target_hidden_vars)

print("model parameters:")
print(model.params)
print("minimum:")
print(model.latent_model(model.hidden_vars, model.params))

model.hidden_vars = target_hidden_vars
results = minimize(model.model,
                   np.ndarray(shape=[2], buffer=np.array([np.pi, np.pi])),
                   method='Newton-CG', jac=model.model_jac, hess=model.model_hes, tol=1E-16
                   )
print(results.message)
print("target model parameters:")
print(results.x)
print("target minimum:")
print(results.fun)


import matplotlib.pyplot as plt

sample_params1 = np.linspace(0, math.pi * 2, 5)
sample_params2 = np.linspace(0, math.pi * 2, 5)
model.hidden_vars = target_hidden_vars
X, Y = meshgrid(sample_params1, sample_params2)
Z = []
for i in range(0, len(X)):
    Z.append(model.observe([X[i], Y[i]]))

figure(1)
fig, ax = subplots()
p = ax.contourf(X, Y, Z)
cb = fig.colorbar(p)
plt.savefig('fig1.png')

with open('data/observation.json', 'r') as fp:
    ZZ = json.load(fp)

Z = []
print(np.max(ZZ))
for i in range(0, 5):
    Z_ = []
    for j in range(0, 5):
        Z_.append(ZZ[i*5 + j])
    Z.append(Z_)

figure(2)
fig, ax = subplots()
p = ax.contourf(X, Y, Z)
cb = fig.colorbar(p)
plt.savefig('fig2.png')

