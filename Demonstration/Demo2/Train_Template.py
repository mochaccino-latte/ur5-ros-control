#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from math import exp


# In[2]:


df = pd.read_excel('demo2_c.xlsx')
joint_select = 6
alpha = 25
alpha_s = 4
n_gaussian = 100
iteration = 80000
lr_rate = 0.005

# In[3]:


q = np.empty([0,1])
qd = np.empty([0,1])
t = df.t.to_numpy()
t = t.reshape((t.shape[0],1))
t = t-t[0][0]
tau = 1/t[len(t)-1]


# In[4]:


pd_q = df.q.to_numpy()
pd_qd = df.qd.to_numpy()
for i in range(len(pd_q)):
    q_t = pd_q[i].replace('(','')
    q_t = q_t.replace(')','')
    q_t = q_t.split(',')
    q_t = np.array(float(q_t[joint_select-1]))
    q = np.vstack((q,q_t))
    qd_t = pd_qd[i].replace('(','')
    qd_t = qd_t.replace(')','')
    qd_t = qd_t.split(',')
    qd_t = np.array(float(qd_t[joint_select-1]))
    qd = np.vstack((qd,qd_t))


# In[5]:


qdd_t = np.vstack((qd[1:],qd[len(qd)-1]))
dt = np.vstack((t[1:],t[len(t)-1]-t[len(t)-2]))-t
dt[len(t)-1] = dt[len(t)-2]
dv = qdd_t-qd
qdd = np.divide(dv,dt)


# In[6]:


q = np.reshape(q,(q.shape[0],))
qd = np.reshape(qd,(qd.shape[0]))
qdd = np.reshape(qdd,(qdd.shape[0]))


# In[7]:
sign = lambda x: 1 if x>=0 else -1

def fref(alpha,q,qd,qdd,tau):
    beta = alpha/4
    g = q[len(q)-1]
    g_vec = np.ones(len(q))*g
    # gmq = g-q[0] if abs(g-q[0]) < 0.00001 else sign(g-q[0])*0.00001
    gmq = g-q[0]
    f_ref = (qdd*(tau**2)-alpha*(beta*(g_vec-q)-qd*tau))/gmq
    return f_ref


# In[8]:


def create_svec(alpha_s,t_vec,tau):
    def update_s(alpha_s,s,t):
        s_t = np.array([exp(-tau*alpha_s*t)])
        s = np.vstack((s,s_t))
        return s
    s = np.array([1])
    for i in range(len(t_vec)-1):
        t = t_vec[i]
        s = update_s(alpha_s,s,t)
    return s


# In[9]:


f_ref = fref(alpha,q,qd,qdd,tau)
print(f_ref)
s = create_svec(alpha_s,t,tau)



# In[11]:


#### Tensorflow Graph ######
with tf.name_scope('s'):
    x = tf.placeholder(tf.float64)
with tf.name_scope('f'):
    f = tf.placeholder(tf.float64)
with tf.name_scope('s_matrix'):
    dim = tf.ones([len(s),n_gaussian],tf.float64)
    s_matrix_t = tf.math.multiply(dim,x)
    s_matrix = tf.transpose(s_matrix_t)
with tf.name_scope('center'):
    c = tf.Variable(tf.linspace(tf.dtypes.cast(0,tf.float64,name=None),1.000,n_gaussian,name = 'center'))
    c = tf.reshape(c,[n_gaussian,1])
with tf.name_scope('c_matrix'):
    dim_c = tf.ones([n_gaussian,len(s)],tf.float64)
    c_matrix = tf.math.multiply(dim_c,c)
with tf.name_scope('bandwidth'):
    h = tf.Variable(0.1*tf.ones((n_gaussian,1),dtype = tf.float64))

smc = s_matrix-c_matrix
h_smc_pow = tf.math.multiply((-h),tf.pow(smc,2))
with tf.name_scope('psi'):
    psi = tf.math.exp(h_smc_pow)

w = tf.Variable(tf.random_normal([n_gaussian,1],name = "weight",dtype = tf.float64))
sum_psi = tf.math.reduce_sum(psi,0)
w_s = tf.math.multiply(w,s_matrix)
wspsi = tf.math.multiply(w_s,psi)
sum_wspsi = tf.math.reduce_sum(wspsi,0)
with tf.name_scope('f_s'):
    f_s = sum_wspsi/sum_psi
    f_s = tf.transpose(f_s)

with tf.name_scope('cost'):
    diff = tf.pow(f-f_s,2)
    cost = tf.reduce_mean(diff,0)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(lr_rate).minimize(cost)


# In[12]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c_p = 10000

    for i in range(iteration):
        _,cost_val = sess.run([optimizer,cost],feed_dict = {x:s,f:f_ref})
        # print('fs={}'.format(fs))
        if abs(cost_val-c_p) <=0.0000000000001:
            break
        c_p = cost_val
        print('iteration {}: {}'.format(i,cost_val))
    w_v,c_v,h_v = sess.run([w,c,h])


# In[ ]:


np.save('Weights/Joint{}/w.npy'.format(joint_select),w_v)
np.save('Weights/Joint{}/c.npy'.format(joint_select),c_v)
np.save('Weights/Joint{}/h.npy'.format(joint_select),h_v)


# In[ ]:





# In[ ]:





# In[ ]:
