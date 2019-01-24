import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.interpolate import RectBivariateSpline
import subprocess
import sys
from sys import stdout
from functools import reduce
from itertools import product
from mpi4py import MPI

worldComm = MPI.COMM_WORLD
mpiRank = worldComm.Get_rank()
mpiSize = worldComm.Get_size()

preprocess_data = False
plot_results = True
load_results = True
extra_filter = True

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
    B1V = np.genfromtxt("./magnetic_1VD.csv", delimiter=",")
    g = np.genfromtxt("./gravity.csv", delimiter=",")
    mines = np.genfromtxt("./something_contained.csv", delimiter=",")

    if mpiRank == 0:
        print("Data loaded! Shapes: %s, %s, %s" % (B.shape, g.shape, mines.shape))

    B[:,1] = np.abs(B[:,1])
    B1V[:,1] = np.abs(B1V[:,1])
    g[:,1] = np.abs(g[:,1])
    mines[:,1] = np.abs(mines[:,1])

    # Normalise data
    for i in range(2):
        xymin = min(B[:,i].min(), g[:,i].min())
        B[:,i] -= xymin
        B1V[:,i] -= xymin
        g[:,i] -= xymin
        mines[:,i] -= xymin

    xymin = [max(B[:,i].min(), g[:,i].min()) for i in range(2)]
    xymax = [max(B[:,i].max(), g[:,i].max()) for i in range(2)]
    mines = np.vstack([mine for mine in mines if (mine[0] <= xymax[0] and mine[1] <= xymax[1] and mine[0] >= xymin[0] and mine[1] >= xymin[1])])

    B[:,2] = (B[:,2] - np.mean(B[:,2]))/np.std(B[:,2])
    B1V[:,2] = (B1V[:,2] - np.mean(B1V[:,2]))/np.std(B1V[:,2])
    g[:,2] = (g[:,2] - np.mean(g[:,2]))/np.std(g[:,2])

    Nx = 1700
    Ny = 1300
    x = np.linspace(xymin[0],xymax[0], Nx)
    y = np.linspace(xymin[1],xymax[1], Ny)

    uBx = unique_tol(B[:,0])
    uBy = unique_tol(B[:,1])
    spB = RectBivariateSpline(uBx, uBy, B[:,2].reshape((len(uBy), len(uBx))).T)

    if mpiRank == 0:
        print("Original mesh sizes (x, y): (%f, %f), new mesh sizes (x, y): (%f, %f)" % (uBx[1]-uBx[0], uBy[1]-uBy[0], (xymax[0] - xymin[0])/Nx, (xymax[1] - xymin[1])/Ny))

    uB1Vx = unique_tol(B1V[:,0])
    uB1Vy = unique_tol(B1V[:,1])
    spB1V = RectBivariateSpline(uB1Vx, uB1Vy, B1V[:,2].reshape((len(uB1Vy), len(uB1Vx))).T)

    ugx = unique_tol(g[:,0])
    ugy = unique_tol(g[:,1])
    spg = RectBivariateSpline(ugx, ugy, g[:,2].reshape((len(ugy), len(ugx))).T)

    BB = spB(x,y)
    gradBB = np.sqrt(spB(x,y,dx=1)**2 + spB(x,y,dy=1)**2)
    BB1V = spB1V(x,y)
    gg = spg(x,y)
    gradgg = np.sqrt(spg(x,y,dx=1)**2 + spg(x,y,dy=1)**2)

    X,Y = np.meshgrid(x,y)

    B_reduced = np.vstack([X.flatten(), Y.flatten(), BB.flatten(), gradBB.flatten()]).T
    B1V_reduced = np.vstack([X.flatten(), Y.flatten(), BB1V.flatten()]).T
    g_reduced = np.vstack([X.flatten(), Y.flatten(), gg.flatten(), gradgg.flatten()]).T

    if mpiRank == 0:
        np.savetxt("reduced_magnetic.csv", B_reduced, delimiter=",")
        np.savetxt("reduced_magnetic1V.csv", B1V_reduced, delimiter=",")
        np.savetxt("reduced_gravity.csv",  g_reduced, delimiter=",")

        np.savez("field_data.npz", B = B_reduced, B1V = B1V_reduced, g = g_reduced, mines = mines)

else:

    #B = np.genfromtxt("reduced_magnetic.csv", delimiter=",")
    #B1V = np.genfromtxt("reduced_magnetic1V.csv", delimiter=",")
    #g = np.genfromtxt("reduced_gravity.csv", delimiter=",")

    data = np.load("field_data.npz")

    B = data["B"]
    B1V = data["B1V"]
    g = data["g"]
    mines = data["mines"]

    x = unique_tol(B[:,0]); Nx = len(x)
    y = unique_tol(B[:,1]); Ny = len(y)

    X = B[:,0].reshape((Nx, Ny))
    Y = B[:,1].reshape((Nx, Ny))

    BB = B[:,2].reshape((Nx, Ny))
    BB1V = B1V[:,2].reshape((Nx, Ny))
    gg = g[:,2].reshape((Nx, Ny))

    gradBB = B[:,3].reshape((Nx, Ny))
    gradgg = g[:,3].reshape((Nx, Ny))


if load_results == False:

    outliers       = np.zeros((Nx, Ny), dtype=bool)
    outliers_score = np.zeros((Nx, Ny))

    k = 60

    def tx(s):
        return min(max(s,0), Nx-1)

    def ty(s):
        return min(max(s,0), Ny-1)

    clf = IsolationForest(behaviour='new', max_samples=0.1,contamination=0.005)

    NNx = np.arange(0,Nx,k)
    NNy = np.arange(0,Ny,k)
    NNp = list(product(NNx, NNy))
    N = len(NNp)
    nprocs  = min(mpiSize,max(N,1))
    NN      = [int(N/nprocs)]*nprocs 
    NN[0]  += N%nprocs
    NN     += [0 for i in range(mpiSize - nprocs)]

    NNcum = np.concatenate([[0], np.cumsum(np.array(NN))])

    bnds = NNcum[mpiRank:mpiRank+2]
    my_Np = NNp[bnds[0]:bnds[1]]

    if mpiRank == 0:
        print("n_local_detections: %d" % len(my_Np))

    if mpiRank == 0:
        stdout.write('\r')
        stdout.write("Outlier detection. Progress: [%-50s] %d%%" % ('', 0))
        stdout.flush()

    for iteration,(i,j) in enumerate(my_Np):
        aux1 = np.arange(tx(i-k), tx(i+k)+1)
        aux2 = np.arange(ty(j-k), ty(j+k)+1)
        T = np.vstack([BB[aux1,:][:,aux2].flatten(), gg[aux1,:][:,aux2].flatten(), gradBB[aux1,:][:,aux2].flatten(), gradgg[aux1,:][:,aux2].flatten(), BB1V[aux1,:][:,aux2].flatten()]).T
        clf.fit(T)
        y_pred = (clf.predict(T).reshape((len(aux1), len(aux2)))-1).astype(bool)
        y_pred_score = clf.score_samples(T).reshape((len(aux1), len(aux2)))
        s = int(k/4)
        outliers[aux1[s]:aux1[-s],aux2[s]:aux2[-s]] |= y_pred[s:-s,s:-s] # logical OR
        outliers_score[aux1[s]:aux1[-s],aux2[s]:aux2[-s]] = np.minimum(outliers_score[aux1[s]:aux1[-s],aux2[s]:aux2[-s]], y_pred_score[s:-s,s:-s])

        if mpiRank == 0:
            stdout.write('\r')
            stdout.write("Outlier detection. Progress: [%-50s] %d%%" % ('='*int(round(50*(iteration+1)/len(my_Np))), int(round(100*(iteration+1)/len(my_Np)))))
            stdout.flush()

    if mpiRank == 0:
        stdout.write('\r%s' % (' '*100))
        stdout.flush()

        stdout.write('\r')

    np.savez("outlier_data_temp%d.npz" % mpiRank, outliers = outliers, outliers_score = outliers_score)

    worldComm.barrier()

    if mpiRank == 0:
        outlier_datas = [np.load("outlier_data_temp%d.npz" % i) for i in range(mpiSize)]
        outliers = reduce(np.maximum, [item["outliers"] for item in outlier_datas])
        outliers_scores = reduce(np.minimum, [item["outliers_score"] for item in outlier_datas])
        np.savez("outlier_data.npz", outliers = outliers, outliers_scores = outliers_scores)

        print("MPI reduction complete")

else:
    outlier_data = np.load("outlier_data.npz")
    outliers = outlier_data["outliers"]
    outliers_scores = outlier_data["outliers_scores"]

if mpiRank == 0 and extra_filter == True:
    #FIXME: probably you do not want to do this more than once
    for i in range(2):
        new_outliers       = np.zeros((Nx, Ny), dtype=bool)
        new_outliers_score = np.zeros((Nx, Ny))
        clf = IsolationForest(behaviour='new', max_samples="auto",contamination=0.15)
        T = np.vstack([BB[outliers], gg[outliers], gradBB[outliers], gradgg[outliers], BB1V[outliers]]).T
        clf.fit(T)
        y_pred = (clf.predict(T)-1).astype(bool)
        y_pred_score = clf.score_samples(T)
        new_outliers[outliers] = y_pred
        outliers = new_outliers.copy()

#plt.cla(); plt.plot(mines[:,0], mines[:,1], "ro"); plt.plot(X[new_outliers.T], Y[new_outliers.T], 'k+')

if mpiRank == 0 and plot_results == True:
    fig = plt.figure(1, figsize=sizes[0][:-1])
    fig.patch.set_facecolor("white")

    plt.subplot(2,3,1)
    plt.contourf(x,y,gg.T)
    plt.title("gravity")

    plt.subplot(2,3,2)
    plt.contourf(x,y,BB.T)
    plt.title("magnetic field")

    plt.subplot(2,3,3)
    plt.contourf(x,y,BB1V.T)
    plt.title("magnetic 1st vert. derivative")

    plt.subplot(2,3,4)
    plt.contourf(x,y,gradgg.T)
    plt.title("gravity gradient norm")

    plt.subplot(2,3,5)
    plt.contourf(x,y,gradBB.T)
    plt.title("magnetic gradient norm")

    plt.subplot(2,3,6)
    if extra_filter == False:
        plt.contourf(x,y,outliers.T, levels=1)
    else:
        plt.contourf(x,y,new_outliers.T, levels=1)

    plt.title("anomalies")

    fig.tight_layout()

    plt.savefig('anomalies.eps', format='eps', dpi=1200)

    plt.subplot(2,3,6)
    plt.contourf(x,y,outliers.T, levels=1)
    plt.plot(mines[:,0], mines[:,1], "ro")
    plt.title("anomalies and mines")
    plt.savefig('anomalies_mines.eps', format='eps', dpi=1200)
