"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
# import torch
import deepxde as dde
import numpy as np
from deepxde.backend import torch

def gen_testdata():
    data = np.load("./dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + dy_x #- 0.2*dy_xx


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
# ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 1, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: 0*x[:,0:1], lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2540*2, num_boundary=80, num_initial=160)
net = dde.nn.FNN([2] + [20] * 5 + [1], "sigmoid", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
# model.train(epochs=3000)
# model.compile("L-BFGS")
losshistory, train_state = model.train(epochs=3000)
#%%
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#%%
X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
# np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))