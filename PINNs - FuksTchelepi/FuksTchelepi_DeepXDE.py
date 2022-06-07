"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
# import torch
import deepxde as dde
import numpy as np
from deepxde.backend import torch
import matplotlib.pyplot as plt
import pickle

def pde_convex(x, y):
    dy_x = dde.grad.jacobian((y**2), x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + dy_x - 0.0025*dy_xx

def pde_concave(x, y):
    M=2;
    dy_x = dde.grad.jacobian(y/(y+(1-y)/M), x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    return dy_t + dy_x

def pde_non_conex(x, y):
    M=1;
    dy_x = dde.grad.jacobian(y**2/(y**2+(1-y**2)/M), x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + dy_x - 0.01*dy_xx

def func(x):
    k_vec=np.array([])
    for i in range(len(x)):
        g=x[i,:]
        if np.isclose(g[0],0):
            k=1
        elif np.isclose(g[0],2):
            k=0
        elif np.isclose(g[1],0):
            k=0
        else:
            k=0
        k_vec=np.append(k_vec,[k],axis=0)
    k_vec=np.reshape(k_vec,(len(k_vec),1))
    return k_vec

#%%
geom = dde.geometry.Interval(0,2)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime,func,lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func,lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde_convex, [bc, ic], num_domain=2000, num_boundary=200, num_initial=100)
net = dde.nn.FNN([2] + [20] * 5 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=3000)
model.compile("L-BFGS")
checkpointer = dde.callbacks.ModelCheckpoint("./model/trained_model", verbose=1, save_better_only=True,period=200)
losshistory, train_state = model.train(epochs=6000,callbacks=[checkpointer])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

#%%
t=0.5;
x=np.linspace(0,1,1000);
xx,tt=np.meshgrid(x, t);
X = np.vstack((np.ravel(xx), np.ravel(tt))).T;
y_pred = model.predict(X);
plt.figure()
plt.plot(X[:,0],y_pred)
