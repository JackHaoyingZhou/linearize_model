#!/usr/bin/env python
import rbdl
import model
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
from ilqr.dynamics import constrain
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from ilqr import iLQR, RecedingHorizonController
from ilqr.controller import RecedingHorizonControllerPath
from ilqr.cost import QRCost, PathQRCost, PathQsRCost, PathQRCostMPC
from ilqr.dynamics import FiniteDiffDynamics
from ilqr.dynamics import AutoDiffDynamics, BatchAutoDiffDynamics, FiniteDiffDynamics
import random
import sys
from tqdm import tqdm

dt = 0.01  # Discrete time-step in seconds.
tf = 2.0
m = 1.0  # Mass in kg.
alpha = 0.1  # Friction coefficient.

my_model = model.dynamic_model()
J_hist = []



def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)

# max_bounds = 10.0
# min_bounds = -10.0


##### 8 and -8
max_bounds = 10.0
min_bounds = -10.0


def f(x, us, i):

    # diff = (max_bounds - min_bounds) / 2.0
    # mean = (max_bounds + min_bounds) / 2.0
    # us = diff * np.tanh(u) + mean
    y = model.runge_integrator(my_model, x, 0.01, us)

    return np.array(y)

def f2(x, u, i):

    y = model.runge_integrator(my_model, x, 0.01, u)

    return np.array(y)
############################################## MODIFY ####################

###### original : return np.array(y)



dynamics = FiniteDiffDynamics(f, 12, 6)

file_name = "/home/jack/backup/leg1.pickle"
#file_name = "/home/jack/catkin_ws/src/ambf_walker/Train/gotozero.pickle"
runner = TPGMMRunner.TPGMMRunner(file_name)

x_path = []
u_path = []

#us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
count = 0
N = runner.get_length()
#print(N)
while count < runner.get_length():
    count += 1
    runner.step()
    u_path.append(runner.ddx.flatten().tolist())
    ##### set it to be random no from -5 to 5 (u_p)
    x = runner.x.flatten().tolist() + runner.dx.flatten().tolist()
    x_path.append(x)

u_path = u_path[:-1]
expSigma = runner.get_expSigma()
size = expSigma[0].shape[0]
Q = [np.zeros((size*2, size*2))]*len(expSigma)
for ii in range(len(expSigma)-2, -1, -1):
    Q[ii][:size, :size] = np.linalg.pinv(expSigma[ii])

x0 = x_path[0]
x_path = np.array(x_path)
u_path = np.array(u_path)

# R = 0.05 * np.eye(dynamics.action_size) # +/-10
R = 5.0e-4 * np.eye(dynamics.action_size)

R[3,3] = 3.0e-3
R[4,4] = 3.0e-3
R[5,5] = 3.0e-3
#R[1,1] = 0.00005
# R[4,4] = 5.0e-6
#
# R[1,1] = 40.0
# R[4,4] = 40.0


# print(x_path)

print(R)

# print(u_path)

us_init = np.random.uniform(-1, 1, (N-1, dynamics.action_size))
#
#
cost = PathQRCost(Q[0], R, x_path=x_path, u_path=u_path)

#print(cost)
#
# # Random initial action path.

#
J_hist = []
ilqr = iLQR(dynamics, cost, N-1)
xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)
# len_us = len(us)
# print(f"len(us)={len_us}")
# print(us)
# print(xs)

#R = 0.01 * np.eye(dynamics.action_size)

#
# R[1,1] = 40.0
# R[4,4] = 40.0

# us = np.where(us>max_bounds,max_bounds,us)
# us = np.where(us<min_bounds,min_bounds,us)

#print(us)

cost2 = PathQRCostMPC(Q[0], R, x_path, us)


ilqr2 = iLQR(dynamics, cost2, N-1)


cntrl = RecedingHorizonControllerPath(x0, ilqr2)

# print(J_hist)
#
# ###### if want to smooth J_hist
#
# plt.figure()
#
# plt.plot(J_hist)
#
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.grid()
#
# plt.show()

plt.ion()

count = 0
fig = plt.figure()
ax1 = fig.add_subplot(321)
ax1.set_xlim([0, 177])
ax1.set_ylim([-0.4, 0.6])
ax2 = fig.add_subplot(323)
ax2.set_xlim([0, 177])
ax2.set_ylim([-0.2, 2.0])
ax3 = fig.add_subplot(325)
ax3.set_xlim([0, 177])
ax3.set_ylim([-0.3, 0.3])
ax4 = fig.add_subplot(322)
ax4.set_xlim([0, 177])
ax4.set_ylim([-0.8, 0.6])
ax5 = fig.add_subplot(324)
ax5.set_xlim([0, 177])
ax5.set_ylim([0.0, 1.7])
ax6 = fig.add_subplot(326)
ax6.set_xlim([0, 177])
ax6.set_ylim([-0.3, 0.3])
x1 = []
y1 = []
x1_follow = []
y1_follow = []
x2 = []
y2 = []
x2_follow = []
y2_follow = []
x3 = []
y3 = []
x3_follow = []
y3_follow = []
x4 = []
y4 = []
x4_follow = []
y4_follow = []
x5 = []
y5 = []
x5_follow = []
y5_follow = []
x6 = []
y6 = []
x6_follow = []
y6_follow = []
us3 = []

line1, = ax1.plot(x1, y1, 'r-')
line2, = ax1.plot(x1_follow, y1_follow, 'b-')
line3, = ax2.plot(x2, y2, 'r-')
line4, = ax2.plot(x2_follow, y2_follow, 'b-')
line5, = ax3.plot(x3, y3, 'r-')
line6, = ax3.plot(x3_follow, y3_follow, 'b-')
line7, = ax4.plot(x4, y4, 'r-')
line8, = ax4.plot(x4_follow, y4_follow, 'b-')
line9, = ax5.plot(x5, y5, 'r-')
line10, = ax5.plot(x5_follow, y5_follow, 'b-')
line11, = ax6.plot(x6, y6, 'r-')
line12, = ax6.plot(x6_follow, y6_follow, 'b-')
count = 0

#print(sys.getsizeof(cntrl.control(us)))


# with open('test.npy', 'wb') as f:
#     np.save(f, us)
# print(len(us))
# print('step1')

for xs2, us2 in tqdm(cntrl.control(us)):
    print(us2[0])
    #print(type(us2[0]))
    us3.append(us2[0])
    x1.append(count)
    y1.append(xs2[0][0])
    x1_follow.append(count)
    y1_follow.append(x_path[count][0])
    x2.append(count)
    y2.append(xs2[0][1])
    x2_follow.append(count)
    y2_follow.append(x_path[count][1])
    x3.append(count)
    y3.append(xs2[0][2])
    x3_follow.append(count)
    y3_follow.append(x_path[count][2])
    x4.append(count)
    y4.append(xs2[0][3])
    x4_follow.append(count)
    y4_follow.append(x_path[count][3])
    x5.append(count)
    y5.append(xs2[0][4])
    x5_follow.append(count)
    y5_follow.append(x_path[count][4])
    x6.append(count)
    y6.append(xs2[0][5])
    x6_follow.append(count)
    y6_follow.append(x_path[count][5])
    print(xs2[0][0:6])
    print(x_path[count][0:6])
    count += 1
    cntrl.set_state(xs2[1])
    line1.set_ydata(y1)
    line1.set_xdata(x1)
    line2.set_ydata(y1_follow)
    line2.set_xdata(x1_follow)
    line3.set_ydata(y2)
    line3.set_xdata(x2)
    line4.set_ydata(y2_follow)
    line4.set_xdata(x2_follow)
    line5.set_ydata(y3)
    line5.set_xdata(x3)
    line6.set_ydata(y3_follow)
    line6.set_xdata(x3_follow)
    line7.set_ydata(y4)
    line7.set_xdata(x4)
    line8.set_ydata(y4_follow)
    line8.set_xdata(x4_follow)
    line9.set_ydata(y5)
    line9.set_xdata(x5)
    line10.set_ydata(y5_follow)
    line10.set_xdata(x5_follow)
    line11.set_ydata(y6)
    line11.set_xdata(x6)
    line12.set_ydata(y6_follow)
    line12.set_xdata(x6_follow)
    fig.canvas.draw()
    fig.canvas.flush_events()
    us = us[1:]
    if count == 177:
        break
# plt.show()
#
# print(us3)
plt.ioff()
# with open('test_1-35.0e-5_4-63.0e-4.npy', 'wb') as f:
#     np.save(f, us3)
plt.show()



# print(len(us))

# print('step2')
# # # file = "/home/jack/catkin_ws/src/linearize_model/test2.npy"
# file = "/home/jack/test.npy"
# with open(file, 'rb') as f:
#     us2 = np.load(f)
#
# us2 = np.where(us2>max_bounds,max_bounds,us2)
# us2 = np.where(us2<min_bounds,min_bounds,us2)
#
# # for i in range(us2.shape[0]):
# #     us2[i] = np.where(us2[i]>max_bounds,max_bounds,us2[i])
# #     us2[i] = np.where(us2[i]<min_bounds,min_bounds,us2[i])
# y = x0
# path = []
# # plt.figure(2)
# for i in range(N-1):
#     y = f2(y, us2[i], i)
#     path.append(y)
#
# path = np.array(path)
#
# f,  axes = plt.subplots(2, 3, sharex=True)
#
# axes[0, 0].plot(x_path[:, 0], "r")
# axes[0, 0].plot(path[:, 0], "g")
# axes[0, 1].plot(x_path[:, 1], "r")
# axes[0, 1].plot(path[:, 1], "g")
# axes[0, 2].plot(x_path[:, 2], "r")
# axes[0, 2].plot(path[:, 2], "g")
#
# axes[1, 0].plot(us2[:, 0], "r")
# axes[1, 1].plot(us2[:, 1], "r")
# axes[1, 2].plot(us2[:, 2], "r")
# plt.show()

# _ = ax1.set_title("Trajectory of Hip")
# _ = ax1.set_ylabel("angle")
# _ = ax1.plot(x_path[:, 2], "r")
# _ = ax1.plot(path[:, 2], "g")
# _ = ax1.legend(["Desired", "actually"])
#
# _ = ax2.set_title("Control Signal")
# _ = ax2.plot(us[:, 2], "g")
# _ = ax2.set_ylabel("Torque")
# _ = ax2.legend(["Control sig"])
# _ = ax2.set_label("Time step")



