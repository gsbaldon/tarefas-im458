"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
# import torch
import deepxde as dde
import numpy as np
from deepxde.backend import torch
import matplotlib.pyplot as plt
import pickle

def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    # dy_xx = dde.grad.jacobian(dy_x, x, i=0, j=0)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_xxx = dde.grad.jacobian(dy_xx, x, i=0, j=0)
    dy_xxxx = dde.grad.jacobian(dy_xxx, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    
    return dy_t + y*dy_x + dy_xx + dy_xxxx

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
geom = dde.geometry.Interval(0,200)
timedomain = dde.geometry.TimeDomain(0, 300)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# bc = dde.icbc.DirichletBC(geomtime,func,lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime,lambda x: np.cos(x[:, 0:1]/16)*(1+np.sin(x[:, 0:1]/16)),lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde, [ic], num_domain=10000, num_boundary=10, num_initial=100)
net = dde.nn.FNN([2] + [20] * 10 + [1], "sin", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=2000)
model.compile("L-BFGS")
checkpointer = dde.callbacks.ModelCheckpoint("./model/trained_model", verbose=1, save_better_only=True,period=200)
losshistory, train_state = model.train(epochs=30000,callbacks=[checkpointer])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

#%%
t=290;
x=np.linspace(0,200,10000);
xx,tt=np.meshgrid(x, t);
X = np.vstack((np.ravel(xx), np.ravel(tt))).T;
y_pred = model.predict(X);
plt.figure()
plt.plot(X[:,0],y_pred)
