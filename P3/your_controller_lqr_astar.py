# Fill in the respective functions to implement the LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
from scipy.signal import StateSpace, lsim, dlsim


class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        self.cum_err = 0
        self.pre_err = 0
        
    def cal_PID(self, delt_T, curr_err, Kp, Ki, Kd):
        df_err = (curr_err - self.pre_err)/delt_T
        self.cum_err+=curr_err * delt_T
        self.pre_err = curr_err
        pid_input = Kp * curr_err + Ki * self.cum_err + Kd * df_err
        return pid_input
        
    def dlqr(self,A,B,Q,R):
        S = np.matrix(linalg.solve_discrete_are(A, B, Q, R))
        K = -np.matrix(linalg.inv(B.T@S@B+R)@(B.T@S@A))
        return K
        
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)
        
        ahead =80 # Ahead mechanism
        _, closest_index = closestNode(X,Y,trajectory)
        if ahead + closest_index >= 6596:
           ahead = 0

        X_des = trajectory[closest_index + ahead,0]
        Y_des = trajectory[closest_index + ahead,1]
        
        psi_des = np.arctan2(Y_des - Y, X_des - X)
        x_vel = 15
        
        # ---------------|Lateral Controller|-------------------------
       
        A = np.array([[0, 1, 0, 0], [0, -4*Ca / (m * xdot), 4*Ca / m, -(2*Ca*(lf - lr))/(m*xdot)], [0, 0, 0, 1], [0, -(2*Ca*(lf - lr)) / (Iz * xdot), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(np.power(lf, 2) + np.power(lr, 2))) / (Iz * xdot)]])
        B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]]) 
        
        C = np.identity(4)
        D = np.array([[0],[0],[0],[0]])
        sys_ct = StateSpace(A, B, C, D)
        sys_dt= sys_ct.to_discrete(delT)
        dt_A = sys_dt.A
        dt_B = sys_dt.B
       
        R=10
        Q=np.array([[1000,0,0,0], [0,100,0,0], [0,0,60,0], [0,0,0,90]])
        K= self.dlqr(dt_A,dt_B,Q,R)
        e1 = 0
        e2 = wrapToPi(psi - psi_des)
        e1dot = ydot + xdot * e2
        e2dot = psidot
        e = np.hstack((e1, e1dot, e2, e2dot))
        delta = float(np.matmul(K,e))
        
        # ---------------|Longitudinal Controller|-------------------------
      
        vel_err = x_vel-xdot
        F = self.cal_PID(delT, vel_err, 200, 0.0001, 0.0001)
        # Return all states and calculated control inputs (F, delta) and obstacle position
        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY