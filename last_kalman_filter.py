import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def system_uncertainty(timestep):
    sigma_a_mat = np.array([ [0.5*(timestep**2)*sigma_a_x],
                             [0.5*(timestep**2)*sigma_a_y],
                             [0.5*(timestep**2)*sigma_a_d],
                             [timestep*sigma_a_x],
                             [timestep*sigma_a_y],
                             [timestep*sigma_a_d], ])
    Q_k = sigma_a_mat.dot(sigma_a_mat.T)
    Q_k = np.diag(np.diag(Q_k))
    return Q_k

def Gaussian_measurement():
    randnoise_x = np.random.normal(mu, rho_x)
    randnoise_y = np.random.normal(mu, rho_y)
    randnoise_d = np.random.normal(mu, rho_d)
    randnoise_v_x = np.random.normal(mu, rho_v_x)
    randnoise_v_y = np.random.normal(mu, rho_v_y)
    randnoise_v_d = np.random.normal(mu, rho_v_d)
    R = np.array([ [rho_x**2, 0, 0, 0, 0, 0],
                   [0, rho_y**2, 0, 0, 0, 0],
                   [0, 0, rho_d**2, 0, 0, 0],
                   [0, 0, 0, rho_v_x**2, 0, 0],
                   [0, 0, 0, 0, rho_v_y**2, 0],
                   [0, 0, 0, 0, 0, rho_v_d**2] ])
    V = np.array([ [randnoise_x],
                   [randnoise_y],
                   [randnoise_d],
                   [randnoise_v_x],
                   [randnoise_v_y],
                   [randnoise_v_d] ])
    return V, R

def prediction(X, timestep):
    randvar_a_x = np.random.normal(mu, sigma_a_x)
    randvar_a_y = np.random.normal(mu, sigma_a_y)
    randvar_a_z = np.random.normal(mu, sigma_a_d)
    F = np.array([ [1, 0, 0, timestep, 0, 0],
                   [0, 1, 0, 0, timestep, 0],
                   [0, 0, 1, 0, 0, timestep],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1] ])
    w = np.array([ [0.5*timestep**2*randvar_a_x],
                   [0.5*timestep**2*(-9.8+randvar_a_y)],
                   [0.5*timestep**2*randvar_a_z],
                   [timestep*randvar_a_x],
                   [timestep*randvar_a_y],
                   [timestep*randvar_a_z], ])
    X_pred = F.dot(X) + w
    return X_pred

def measurement(X_measurement, H, V):
    Z = H.dot(X_measurement) + V
    return  Z

def prediction_system_uncertainty(P, Q_k, timestep):
    F = np.array([ [1, 0, 0, timestep, 0, 0],
                   [0, 1, 0, 0, timestep, 0],
                   [0, 0, 1, 0, 0, timestep],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1] ])
    P_p = np.diag(np.diag(F.dot(P).dot(F.T))) + Q_k
    return P_p

def kalman_filter(X, Y, H, P, timestep):
    # print('X_input %s (timestep %s)' %(X,timestep))
    # print('H %s' %H)
    # print('P %s' %P)
    I = np.identity(6, dtype=float)
    # H = np.identity(6, dtype=float)
    # P = np.zeros((6,6))
    X_pred = prediction(X, timestep)  # Prediction steps
    # print('X_pred %s' %X_pred)
    Q_k = np.diag(np.diag(system_uncertainty(timestep)))  # System uncertainty that from prediction steps
    # print('Q_k %s' %Q_k)
    V, R = Gaussian_measurement()   # Noise Vector (V) and Noise Covariance Matrix (R)
    # print('R %s' %R)
    Z = measurement(Y, H, V)   # Observation Matrix
    # print('Z %s' %Z)
    # print('Before P %s' %P)
    P_p = prediction_system_uncertainty(P, Q_k, timestep)
    # print('After P %s' %P_p)
    S = H.dot(P_p).dot(H.T) + R
    K = P_p.dot(H.T).dot(inv(S))
    X = X_pred + K.dot(Z - H.dot(X_pred))
    P = (I - K.dot(H)).dot(P_p)
    return X, P, X_pred

# timestep = 0.025
mu = 0
sigma_a_x = 0.5
sigma_a_y = 0.5
sigma_a_d = 0.5
rho_x = 0.005
rho_y = 0.005
rho_d = 0.008
rho_v_x = 0.05
rho_v_y = 0.05
rho_v_d = 0.05

# No Ping-Pong Detection
def init_state(X_0, Y, timestep):
    P_0 = np.array([ [sigma_x**2, 0, 0, 0, 0, 0],
                   [0, sigma_y**2, 0, 0, 0, 0],
                   [0, 0, sigma_d**2, 0, 0, 0],
                   [0, 0, 0, sigma_v_x**2, 0, 0],
                   [0, 0, 0, 0, sigma_v_y**2, 0],
                   [0, 0, 0, 0, 0, sigma_v_d**2] ])
    H_0 = np.zeros((6,6))
    X_kal, P, X_pred = kalman_filter(X_0, Y, H_0, P_0, timestep)
    # print('X_kal %s' %(X_kal))
    # print('P %s' %(P))
    return X_kal, P, X_pred

def init_detection_state(X, P, Y, timestep):
    # print('Timestep %s' %timestep)
    H = np.array([ [1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0] ])
    X_kal, P, X_pred = kalman_filter(X, Y, H, P, timestep)
    # print('X_kal %s' %(X_kal))
    # print('P %s' %(P))
    return X_kal, P, X_pred

sigma_x = 100
sigma_y = 100
sigma_d = 100
sigma_v_x = 100
sigma_v_y = 100
sigma_v_d = 100
