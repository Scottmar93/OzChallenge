import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.interpolate import RectBivariateSpline
import subprocess
import sys

preprocess_data = True

# change the round factor if you like
r = 1
sizes = []
screens = [l.split()[-3:] for l in subprocess.check_output(["xrandr"]).decode("utf-8").strip().splitlines() if " connected" in l]
for s in screens:
    w = float(s[0].replace("mm", "")); h = float(s[2].replace("mm", "")); d = ((w**2)+(h**2))**(0.5)
    sizes.append([2*round(n/25.4, r) for n in [w, h, d]])

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
plt.rc('font', family='serif', size=22)
plt.rc('xtick', labelsize=24)     
plt.rc('ytick', labelsize=24)

plt.ion()


def unique_tol(A):
    _,idx = np.unique(A.round(decimals=9), return_index=True)
    return A[idx]


if preprocess_data == True:

    # load data
    B = np.genfromtxt("./magnetic.csv", delimiter=",")
    g = np.genfromtxt("./gravity.csv", delimiter=",")
    mines = np.genfromtxt("./mines.csv", delimiter=",")

    print("Data loaded! Shapes: %s, %s, %s" % (B.shape, g.shape, mines.shape))

    B[:,1] = np.abs(B[:,1])
    g[:,1] = np.abs(g[:,1])
    mines[:,1] = np.abs(mines[:,1])

    # Normalise data
    for i in range(2):
        xymin = min(B[:,i].min(), g[:,i].min())
        B[:,i] -= xymin
        g[:,i] -= xymin
        mines[:,i] -= xymin

    xymax = [max(B[:,i].max(), g[:,i].max()) for i in range(2)]
    mines = np.vstack([mine for mine in mines if mine[0] <= xymax[0] and mine[1] <= xymax[1]])

    B[:,2] = (B[:,2] - np.mean(B[:,2]))/np.std(B[:,2])
    g[:,2] = (g[:,2] - np.mean(g[:,2]))/np.std(g[:,2])

    Nx = 600
    Ny = 600
    x = np.linspace(xymax[0]/3,2/3*xymax[0], Nx)
    y = np.linspace(xymax[1]/3,2/3*xymax[1], Ny)

    uBx = unique_tol(B[:,0])
    uBy = unique_tol(B[:,1])
    spB = RectBivariateSpline(uBx, uBy, B[:,2].reshape((len(uBy), len(uBx))).T)

    print("Original mesh sizes (x, y): (%f, %f), new mesh sizes (x, y): (%f, %f)" % (uBx[1]-uBx[0], uBy[1]-uBy[0], xymax[0]/3/Nx, xymax[1]/3/Ny))

    ugx = unique_tol(g[:,0])
    ugy = unique_tol(g[:,1])
    spg = RectBivariateSpline(ugx, ugy, g[:,2].reshape((len(ugy), len(ugx))).T)

    BB = spB(x,y)
    gradBB = np.sqrt(spB(x,y,dx=1)**2 + spB(x,y,dy=1)**2)
    gg = spg(x,y)
    gradgg = np.sqrt(spg(x,y,dx=1)**2 + spg(x,y,dy=1)**2)

    X,Y = np.meshgrid(x,y)

    B_reduced = np.vstack([X.flatten(), Y.flatten(), BB.flatten(), gradBB.flatten()]).T
    g_reduced = np.vstack([X.flatten(), Y.flatten(), gg.flatten(), gradgg.flatten()]).T

    np.savetxt("reduced_magnetic.csv", B_reduced, delimiter=",")
    np.savetxt("reduced_gravity.csv",  g_reduced, delimiter=",")

else:

    B = np.genfromtxt("reduced_magnetic.csv", delimiter=",")
    g = np.genfromtxt("reduced_gravity.csv", delimiter=",")

    print("Data loaded!")

    x = unique_tol(B[:,0]); Nx = len(x)
    y = unique_tol(B[:,1]); Ny = len(y)

    X = B[:,0].reshape((Nx, Ny))
    Y = B[:,1].reshape((Nx, Ny))

    BB = B[:,2].reshape((Nx, Ny))
    gg = g[:,2].reshape((Nx, Ny))

    gradBB = B[:,3].reshape((Nx, Ny))
    gradgg = g[:,3].reshape((Nx, Ny))


outliers = np.zeros((Nx, Ny), dtype=bool)

k = 30

def tx(s):
    return min(max(s,0), Nx-1)

def ty(s):
    return min(max(s,0), Ny-1)

#clf = LocalOutlierFactor(n_neighbors=(2*k+1)**2, contamination = 0.00001)
clf = IsolationForest(behaviour='new', max_samples=0.1,contamination=0.005)

for i in range(0,Nx,k):
    for j in range(0,Ny,k):
        aux1 = np.arange(tx(i-k), tx(i+k)+1)
        aux2 = np.arange(ty(j-k), ty(j+k)+1)
        #T = np.vstack([BB[aux1,:][:,aux2].flatten(), gg[aux1,:][:,aux2].flatten()]).T
        T = np.vstack([BB[aux1,:][:,aux2].flatten(), gradBB[aux1,:][:,aux2].flatten(), gg[aux1,:][:,aux2].flatten(), gradgg[aux1,:][:,aux2].flatten()]).T
        y_pred = (clf.fit_predict(T).reshape((len(aux1), len(aux2)))-1).astype(bool)
        s = int(k/4)
        outliers[aux1[s]:aux1[-s],aux2[s]:aux2[-s]] |= y_pred[s:-s,s:-s] # logical OR
    print("ciao")


fig = plt.figure(1, figsize=sizes[0][:-1])
fig.patch.set_facecolor("white")

plt.subplot(2,2,1)
plt.contourf(x,y,gg.T)

plt.subplot(2,2,2)
plt.contourf(x,y,BB.T)

plt.subplot(2,2,3)
plt.contourf(x,y,gradgg.T)

plt.subplot(2,2,4)
plt.contourf(x,y,gradBB.T)

fig = plt.figure(2, figsize=sizes[0][:-1])
fig.patch.set_facecolor("white")
plt.contourf(x,y,outliers.T, levels=1)
