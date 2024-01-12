# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
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

        # Add additional member variables according to your need here.
        
        self.vel_pre_err = 0
        self.vel_cum_err = 0

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        _, closest_node = closestNode(X, Y, trajectory)

        time_ahead = 15

        if (closest_node + time_ahead >= trajectory.shape[0]):
            time_ahead = 0

        # Retrieve the target values corresponding to the nearest trajectory point.
        
        V_des = 18
        X_des = trajectory[closest_node + time_ahead, 0]
        Y_des = trajectory[closest_node + time_ahead, 1]
        psi_des = np.arctan2(Y_des - Y, X_des - X)

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta).

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        V_x = xdot
        A = np.array([[0, 1, 0, 0],
        [0, -4*Ca / (m * V_x), 4*Ca/m, -(2*Ca*(lf - lr))/(m*V_x)],
        [0, 0, 0, 1],
        [0, -(2*Ca*(lf - lr)) / (Iz * V_x), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(lf*lf + lr * lr)) / (Iz * V_x)]])
        B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
        
        des_poles = np.array([-5,-4,-0.5,0])
        K = signal.place_poles(A, B, des_poles).gain_matrix

        e1 = (np.power(np.power(X_des - X, 2) + np.power(Y_des - Y, 2), 0.5))
        e2 = wrapToPi(psi - psi_des)
        e1_dot = -ydot * np.sin(e2) + xdot * np.cos(e2)
        e2_dot = psidot

        e = np.hstack((e1, e1_dot, e2, e2_dot)).reshape(4,1)
        delta = wrapToPi(np.dot(-K,e)[0,0])


        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        vel_err = V_des-xdot
        vel_df_err = (vel_err-self.vel_pre_err)/delT
        self.vel_cum_err += vel_err * delT
        self.vel_pre_err = vel_err

        kp2 = 40
        ki2 = 0.0001
        kd2 = 0.0001
        F = kp2*vel_err+ki2*self.vel_cum_err+kd2*vel_df_err
        if(F < 0):
            F = 0
        elif(F > 15736):
            F = 15736

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
