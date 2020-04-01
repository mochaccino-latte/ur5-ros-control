import tensorflow as tf
import numpy as np
from scipy.integrate import odeint
from math import exp, pi,sqrt

def canoSystem(tau,t):
    alpha_s = 4
    s = exp(-tau*alpha_s*t)
    return s

def dmp(g,q,qd,tau,s,q0,W,Name = "DMP"):
    alpha = tf.constant(25,dtype=tf.float64)
    beta = alpha/4
    w,c,h = W
    n_gaussian = w.shape[0]
    with tf.name_scope(Name):
        w_tensor = tf.constant(w,dtype=tf.float64,name='w')
        c_tensor = tf.constant(c,dtype=tf.float64,name='c')
        h_tensor = tf.constant(h,dtype=tf.float64,name='h')

        with tf.name_scope('s'):
            s_tensor = s*tf.ones(n_gaussian,dtype=tf.float64)
        smc_pow = tf.pow(s_tensor-c_tensor,2)
        h_smc_pow = tf.math.multiply(smc_pow,(-h_tensor))
        with tf.name_scope('psi'):
            psi = tf.math.exp(h_smc_pow)
        sum_psi = tf.math.reduce_sum(psi,0)
        wpsi = tf.math.multiply(w_tensor,psi)
        wpsis = tf.math.reduce_sum(wpsi*s,0)
        with tf.name_scope('fs'):
            fs =wpsis/sum_psi
        qdd = alpha*(beta*(g-q)-tau*qd)+fs*(g-q0)
    return qdd

##### Final Catesian Position of Demonstration) #####
demo_x = np.array([-8.15926729e-01, -0.75961731, -0.3964087, -0.29553788, -0.04094927, -0.14693912, -0.41827111, -8.16843140e-01, -0.09284764, -0.57153495, -0.67251442, -0.36517125, -7.62308039e-01, -0.78029185, -6.57512038e-01])
demo_y = np.array([-2.96043917e-01, -0.18374539, 0.6690932,  0.21733157,  0.78624892,  0.7281835,   -0.66857267, -2.92201916e-01, -0.77947085, -0.28442803, 0.36890422,  -0.41997883, -1.20031233e-01, -0.19321253, -1.05877890e-01])
demo_z = np.array([-3.97988321e-03, 0.35300285,  0.13734106, 0.1860831,   0.06178831,  0.06178831,  0.10958549,  -5.64177448e-03, 0.0383235,   0.33788756,  0.30410704,  0.47738503,  8.29937352e-03,  0.17253172,  3.62063583e-01])



#### Input Tensors ####

 ## Common Input ##
s = tf.placeholder(tf.float64,name='s')
tau = tf.placeholder(tf.float64,name='tau')

xg = tf.placeholder(tf.float64,name='xg')
yg = tf.placeholder(tf.float64,name='yg')
zg = tf.placeholder(tf.float64,name='zg')

 ## joints ##
g = (tf.placeholder(tf.float64,name='g1'),
    tf.placeholder(tf.float64,name='g2'),
    tf.placeholder(tf.float64,name='g3'),
    tf.placeholder(tf.float64,name='g4'),
    tf.placeholder(tf.float64,name='g5'),
    tf.placeholder(tf.float64,name='g6'))

q = (tf.placeholder(tf.float64,name='q1'),
    tf.placeholder(tf.float64,name='q2'),
    tf.placeholder(tf.float64,name='q3'),
    tf.placeholder(tf.float64,name='q4'),
    tf.placeholder(tf.float64,name='q5'),
    tf.placeholder(tf.float64,name='q6'))

qd = (tf.placeholder(tf.float64,name='qd1'),
    tf.placeholder(tf.float64,name='qd2'),
    tf.placeholder(tf.float64,name='qd3'),
    tf.placeholder(tf.float64,name='qd4'),
    tf.placeholder(tf.float64,name='qd5'),
    tf.placeholder(tf.float64,name='q06'))

q0 = (tf.placeholder(tf.float64,name='q01'),
    tf.placeholder(tf.float64,name='q02'),
    tf.placeholder(tf.float64,name='q03'),
    tf.placeholder(tf.float64,name='q04'),
    tf.placeholder(tf.float64,name='q05'),
    tf.placeholder(tf.float64,name='q06'))

#### Movement Library #####
dmps = [{},{},{},{},{},{}]
for i in range(15):
 path = 'Demonstration/Demo{}/Weights/'.format(i+1)
 for j in range(6): ### j = joint number
     path_j = path+'joint{}/'.format(j+1)
     w = np.load(path_j+'w.npy')
     c = np.load(path_j+'c.npy')
     h = np.load(path_j+'h.npy')
     W = (w,c,h)
     # def dmp(g,q,qd,tau,s,q0,W,Name = "DMP"):
     dmps[j]['{}_{}'.format(j+1,i+1)] = dmp(g[j], q[j], qd[j], tau, s, q0[j], W, Name="DMP{}_{}".format(j+1,i+1))


#### Contributin Functions ####
with tf.name_scope("Con"):
    xg_ref = tf.constant(demo_x, dtype=tf.float64,name="x_con")
    yg_ref = tf.constant(demo_y, dtype=tf.float64,name="y_con")
    zg_ref = tf.constant(demo_z, dtype=tf.float64,name="z_con")

    xg2 = tf.pow(xg_ref-xg, 2)
    yg2 = tf.pow(yg_ref-yg, 2)
    zg2 = tf.pow(zg_ref-zg, 2)

    sum = xg2+yg2+zg2
    con = 1.9947114020071635 * tf.math.exp(-0.5*sum/0.4472135954999579) # Normal Distribution
    re = tf.reduce_sum(con, axis=0)

#### Gating Network #####
dmp_joint = []
dmpNet = []
for i in range(len(dmps)):
    values = list(dmps[i].values())
    joint = tf.concat(values, axis=0)
    with tf.name_scope('DMPNet{}'.format(i+1)):
        dmpNet_i = tf.reduce_sum(tf.math.multiply(joint,con),axis=0)/tf.reduce_sum(con, axis=0)
    dmpNet.append(dmpNet_i)





### Tensorflow Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    def dynamics(x,t,tau_v,g_v,q0_v,position,sess):
        s_v = canoSystem(tau_v,t)

        feeddict = {g[0]:g_v[0],g[1]:g_v[1],g[2]:g_v[2],g[3]:g_v[3],g[4]:g_v[4],g[5]:g_v[5],
                    q[0]:x[0],q[1]:x[1],q[2]:x[2],q[3]:x[3],q[4]:x[4],q[5]:x[5],
                    qd[0]:x[6],qd[1]:x[7],qd[2]:x[8],qd[3]:x[9],qd[4]:x[10],qd[5]:x[11],
                    q0[0]:q0_v[0],q0[1]:q0_v[1],q0[2]:q0_v[2],q0[3]:q0_v[3],q0[4]:q0_v[4],q0[5]:q0_v[5],
                    tau:tau_v,s:s_v,xg:position[0],yg:position[1],zg:position[2]
                    }

        qdd1_v,qdd2_v,qdd3_v,qdd4_v,qdd5_v,qdd6_v = sess.run(dmpNet,feed_dict = feeddict)
        dx = [x[6],x[7],x[8],x[9],x[10],x[11],qdd1_v,qdd2_v,qdd3_v,qdd4_v,qdd5_v,qdd6_v]
        return dx
    t = np.linspace(0, 1.423553944, 100)
    tau_v = float(1)/1.423553944
    q0_v = [-0.0003235975848596695, -1.040771786366598, 1.6213598251342773, -0.34193402925600225, 1.5711277723312378, 3.141711950302124]
    v0 = [0,0,0,0,0,0]
    g_v = [-0.4201243559466761, -1.3455780188189905, 1.6121912002563477, -0.055014912282125294, 1.2821934223175049, 3.1416163444519043]
    x0 = []
    x0.extend(q0_v)
    x0.extend(v0)
    # print(q0_v)
    q = odeint(dynamics,x0,t,args=(tau_v,g_v,q0_v,position,sess))
    np.save('q.npy',q)
