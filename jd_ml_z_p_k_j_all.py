#!/usr/bin/env python
# coding: utf-8

# In[2]:


from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp



def get_angle_degrees(v1, v2):
    # 计算两个向量的点积
    dot_product = np.dot(v1, v2)

    # 计算两个向量的模长
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    # 计算两个向量的夹角（弧度）
    angle_in_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

    # 将弧度转换为角度
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees


def to_unit(v):
    # 定义向量v
    v 
    # 计算向量v的模长
    magnitude = np.linalg.norm(v)

    # 计算向量v的单位向量
    unit_vector = v / magnitude

    return unit_vector



# initial : 
# uav_loc = [0,0]

# R
# view_theta_r0
# => obj_loc = [R*sin(view_theta_r0), R*cos(view_theta_r0)]
 
# v_obj = [1,-2]   => v_obj_unit 

# speed_obj = 0.2

# for step_cnt in range(2000):
#     obj_loc + v_obj_unit * speed_obj
#     print (obj_loc) 
# view_theta_new = acos( []

### global var ###
global uav_loc
global R
global view_theta_r0
global v_obj
global speed_obj
global max_range

global arr_out

uav_loc = np.array([0,0])
R=100
v_obj = np.array([1,-3])
speed_obj = 1
view_theta_r0 = 3
max_range = 512

##################


def gen_array_with_args(idx, uav_loc, R, v_obj, speed_obj, view_theta_r0, max_range):
    global arr_out
    
    arr_out[idx,0]=idx
    arr_out[idx,1:1+2]=uav_loc
    arr_out[idx,3] =R
    arr_out[idx,4:4+2] = v_obj
    arr_out[idx,6]= speed_obj
    arr_out[idx,7]= view_theta_r0
    
    
    v_obj=to_unit(v_obj)
    view_theta_r0_rad = np.deg2rad(view_theta_r0)
    obj_loc = np.array([R*sin(view_theta_r0_rad), R*cos(view_theta_r0_rad)])

    

    obj_loc_set = np.zeros((max_range,2))
    view_theta_degree_set = np.zeros((max_range,2))
    
    for step_cnt in range(max_range):
        obj_loc = obj_loc + v_obj * speed_obj
        obj_loc_set[step_cnt,:] = obj_loc
        
        view_theta_degree = get_angle_degrees([0,1], obj_loc-uav_loc)
        view_theta_degree_set[step_cnt,:] = [view_theta_degree,step_cnt]

#     disp(obj_loc_set)

    
#     plt.scatter(obj_loc_set[:,0], obj_loc_set[:,1])
#     plt.scatter(view_theta_degree_set[:,0], view_theta_degree_set[:,1])
    
    arr_out[idx,8:8+512] = view_theta_degree_set[:,0]
    
    
    
#     plt.show()
 



#main_ 
def main():
    global uav_loc #观察位置
    global R   #半径
    global view_theta_r0  #角度
    global v_obj  #潜艇方向
    global speed_obj #潜艇速度
    global max_range  #观察次数
    
    v_obj=to_unit(v_obj)
    view_theta_r0_rad = np.deg2rad(view_theta_r0)
    obj_loc = np.array([R*sin(view_theta_r0_rad), R*cos(view_theta_r0_rad)])

    

    obj_loc_set = np.zeros((max_range,2))
    view_theta_degree_set = np.zeros((max_range,2))
    
    for step_cnt in range(max_range):
        obj_loc = obj_loc + v_obj * speed_obj
        obj_loc_set[step_cnt,:] = obj_loc
        
        view_theta_degree = get_angle_degrees([0,1], obj_loc-uav_loc)
        view_theta_degree_set[step_cnt,:] = [view_theta_degree,step_cnt]

#     disp(obj_loc_set)

    
#     plt.scatter(obj_loc_set[:,0], obj_loc_set[:,1])
    plt.scatter(view_theta_degree_set[:,0], view_theta_degree_set[:,1])
    
    plt.show()


# uav_loc = 

arr_out= np.zeros([50000,8+512], dtype=np.float32)



print(uav_loc.shape[0])



# main()
idx = 0
for v_obj_ in np.array([[1,-3], [1,-3.3], [1,-4], [1,-6], [1.3,-5.1]]):
    for speed_obj_ in np.linspace(1,7,10):
        speed_obj = speed_obj_
        for R_ in np.linspace(150,500,10):
            R = R_
            for theta_ in np.linspace(1,77,100):
                view_theta_r0 = theta_
                if(idx%1000==0):
                    print("- idx is ", idx)
                
#                 gen_array_with_args(idx, uav_loc, R, v_obj, speed_obj, view_theta_r0, max_range)
                idx = idx+1
                
# arr_out[0,:]
# np.save("./cpa.npy", arr_out)



# In[36]:


from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp

arr_in = np.load("./cpa.npy")

arr_in_cpa = arr_in[:, 8:]



def get_idx_data(idx, arr_in):
    max_idx = 512
    e_view_theta = arr_in[idx,8:8+max_idx]
    return e_view_theta


def show_idx_figure(idx, arr_in):
    e_view_theta = get_idx_data(idx,arr_in)
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()



def show_idx_figure_cpa(idx, arr_in_cpa):
    e_view_theta = arr_in_cpa[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()

# show_idx_figure(2, arr_in)
# show_idx_figure(20000, arr_in)
show_idx_figure_cpa(4000,arr_in_cpa)


arr_in_cpa.shape
# np.random.random(arr_in_cpa.shape)








# In[23]:


#  for ma to try 
import tensorflow as tf

from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp

arr_in = np.load("./cpa.npy")



arr_in_cpa = arr_in[:, 8:]
sigma_noise = np.max(arr_in_cpa[0:10000])/10/2.5




def get_idx_data(idx, arr_in):
    max_idx = 512
    e_view_theta = arr_in[idx,8:8+max_idx]
    return e_view_theta


    
def show_idx_figure(idx, arr_in):
    e_view_theta = get_idx_data(idx,arr_in)
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()



def show_idx_figure_cpa(idx, arr_in_cpa):
    e_view_theta = arr_in_cpa[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()

# show_idx_figure(2, arr_in)
# show_idx_figure(20000, arr_in)
# show_idx_figure_cpa(1,arr_in_cpa)

def create_noise_mat(sigma_noise, arr_in_cpa):
    noise = np.random.uniform(-1*sigma_noise, 1*sigma_noise, arr_in_cpa.shape)
    return noise
    


noise = create_noise_mat(sigma_noise, arr_in_cpa)

arr_in_cpa_noise = arr_in_cpa+noise

show_idx_figure_cpa(1,arr_in_cpa)
show_idx_figure_cpa(1,arr_in_cpa_noise)


np.save("./raw_cpa.npy", arr_in_cpa)
np.save("./raw_cpa_noise.npy", arr_in_cpa_noise) 











# In[2]:


# train cpa_noise to cpa 
import tensorflow as tf
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from sklearn.model_selection import train_test_split


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(8,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])


def reshape_2_timesteps_feature(data_2d, time_steps, features):
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d

def show_idx_figure_cpa(idx, arr_in_cpa):
    e_view_theta = arr_in_cpa[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()  
    

flag_train = 1
epoches = 100

cpa_noise = np.load("./data/raw_cpa_noise.npy")

raw_cpa = np.load("./data/raw_cpa.npy")

time_steps = 64
features = int(raw_cpa.shape[1]/time_steps)




x_train, x_test, y_train, y_test  = train_test_split(cpa_noise, raw_cpa, test_size=0.2)

display(x_train.shape)




model = tf.keras.Sequential(
[

    tf.keras.layers.LSTM(32, return_sequences=True, input_shape= (time_steps, features) ), 
    tf.keras.layers.Dropout(0.22),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(y_train.shape[1])

]

)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])



history = model.fit(reshape_2_timesteps_feature(x_train,time_steps, features), y_train, epochs=epoches, batch_size=512 , callbacks=[early_stop])

y_pred = model.predict(reshape_2_timesteps_feature(x_test,time_steps, features))

loss = model.evaluate(reshape_2_timesteps_feature(x_test,time_steps, features), y_test)

# model.save("./model_cpa/cpa.h5")








# In[ ]:


# inference cpa_noise => cpa 
model_new=tf.keras.models.load_model("./model_cpa/cpa.h5")
model_new.summary()

y_test_ = model_new.predict(reshape_2_timesteps_feature(x_test[0:1000], time_steps, features))

y_test_.shape


colors = ['r', 'g', 'b']
markers = ['o', '^', 's']




show_idx_figure_cpa(333, x_test)
show_idx_figure_cpa(333, y_test_)
show_idx_figure_cpa(333, y_test)


# In[1]:


#  for ma to try 
import tensorflow as tf
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from sklearn.model_selection import train_test_split


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(8,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])


def reshape_2_timesteps_feature(data_2d, time_steps, features):
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d

def show_idx_figure_cpa(idx, arr_in_cpa):
    e_view_theta = arr_in_cpa[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()  

def show_idx_figure_cpa_rate(idx, arr_in_cpa_rate):
    e_view_theta_rate = arr_in_cpa_rate[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, e_view_theta_rate)
    plt.show()  

flag_train = 0
epoches = 100

cpa_noise = np.load("./data/raw_cpa_noise.npy")

raw_cpa = np.load("./data/raw_cpa.npy")


row_rate,col_rate = np.gradient(raw_cpa)



raw_cpa_rate = np.log10(np.abs(col_rate))+ 0.5

# display(raw_cpa_rate.shape)


time_steps = 64
features = int(raw_cpa.shape[1]/time_steps)


# cpa_noise => cpa 
x_train_raw, x_test_raw, y_train_raw, y_test_raw  = train_test_split(cpa_noise, raw_cpa, test_size=0.2)

# cpa_noise => cpa_rate 
x_train, x_test, y_train, y_test  = train_test_split(cpa_noise, raw_cpa_rate, test_size=0.2)

y_train_raw_rate = y_train
y_test_raw_rate = y_test

display(x_train.shape)


if flag_train:

    model = tf.keras.Sequential(
    [

        tf.keras.layers.LSTM(32, return_sequences=True, input_shape= (time_steps, features) ), 
        tf.keras.layers.Dropout(0.22),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(y_train.shape[1])

    ]

    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])



    history = model.fit(reshape_2_timesteps_feature(x_train,time_steps, features), y_train, epochs=epoches, batch_size=512 , callbacks=[early_stop])

    y_pred = model.predict(reshape_2_timesteps_feature(x_test,time_steps, features))

    loss = model.evaluate(reshape_2_timesteps_feature(x_test,time_steps, features), y_test)

    model.save("./model_cpa/cpa_rate.h5")


else:
    model_cpa = tf.keras.models.load_model("./model_cpa/cpa.h5")
    model_cpa_rate=tf.keras.models.load_model("./model_cpa/cpa_rate.h5")
    
    model_cpa.summary()
    model_cpa_rate.summary()
    
    y_test_cpa_ = model_cpa.predict(reshape_2_timesteps_feature(x_test_raw[0:3000], time_steps, features))
    y_test_cpa_rate_ = model_cpa_rate.predict(reshape_2_timesteps_feature(x_test[0:3000], time_steps, features))

    y_test_cpa_rate_.shape

x = np.linspace(0,512,512)
x = np.int32(x)


colors = ['r', 'g', 'b']
markers = ['.', '.', 's']

for idx in range(2,2999,259):
# idx = 622




    plt.scatter(y_test_cpa_[idx], x, c=colors[0], marker=markers[0] )
    plt.scatter(y_test_raw[idx], x, c=colors[1], marker=markers[0] )
    plt.show()

    plt.scatter(y_test_cpa_[idx], y_test_cpa_rate_[idx], c=colors[0], marker=markers[0])
    plt.scatter(y_test_raw[idx], y_test_raw_rate[idx], c=colors[1], marker=markers[1])
    plt.show()






# In[123]:


colors = ['r', 'g', 'b']
markers = ['.', '.', 's']

x = np.linspace(0,512,512)
x = np.int32(x)

y = y_test
y_ = y_test_

idx = 77



np.gradient(y[idx])
np.gradient(y_[idx])

plt.scatter(y[idx],x, c=colors[0], marker=markers[0])
plt.scatter(y_[idx],x, c=colors[1], marker=markers[1])
plt.show()




plt.scatter(y[idx], np.log10(abs(np.gradient(y[idx])))+ 0.5,     c=colors[0], marker=markers[0])
plt.scatter( y_[idx], np.log10(abs(np.gradient(y_[idx])))+ 0.5,  c=colors[1], marker=markers[1])
plt.show()



# In[20]:


from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp


cpa_arr = np.load("./data/cpa.npy")

np.random.shuffle(cpa_arr)
# np.save("./data/cpa_rnd.npy", cpa_arr)  



# In[22]:


from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp

def create_noise_mat(sigma_noise, arr_in_cpa):
    noise = np.random.uniform(-1*sigma_noise, 1*sigma_noise, arr_in_cpa.shape)
    return noise
    


cpa_arr = np.load("./data/cpa_rnd.npy")

cpa_arr_raw = cpa_arr[:, 8:]


sigma_noise = np.max(cpa_arr_raw[0:10000])/10/2.6

noise = create_noise_mat(sigma_noise, cpa_arr_raw)

cpa_arr_raw_noise = cpa_arr_raw + noise 

# np.save("./data/cpa_raw_rnd.npy", cpa_arr_raw)
# np.save("./data/cpa_raw_noise_rnd.npy", cpa_arr_raw_noise)

print("save ok")




# In[79]:


# show the data of cpa noise , cpa
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp

def show_idx_figure_cpa(idx, arr_in_cpa):
    e_view_theta = arr_in_cpa[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr, s=1)
    plt.show()

    

cpa_arr_raw = np.load("./data/cpa_raw_rnd.npy")

cpa_arr_raw_noise = np.load("./data/cpa_raw_noise_rnd.npy")

idx = 111
show_idx_figure_cpa(idx, cpa_arr_raw_noise)
show_idx_figure_cpa(idx, cpa_arr_raw)





# In[45]:


# train: cpa noise => cpa
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp

def reshape_2_timesteps_feature(data_2d, time_steps, features):
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d


    
cpa_noise = np.load("./data/cpa_raw_noise_rnd.npy")
raw_cpa = np.load("./data/cpa_raw_rnd.npy")


epoches = 100
time_steps = 64
features = int(raw_cpa.shape[1]/time_steps)

x_train, x_test, y_train, y_test  = train_test_split(cpa_noise, raw_cpa, test_size=0.2)

display(x_train.shape)


model = tf.keras.Sequential(
[

    tf.keras.layers.LSTM(32, return_sequences=True, input_shape= (time_steps, features) ), 
    tf.keras.layers.Dropout(0.22),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(y_train.shape[1])

]

)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])


history = model.fit(reshape_2_timesteps_feature(x_train,time_steps, features), y_train, epochs=epoches, batch_size=512 , callbacks=[early_stop])

# y_pred = model.predict(reshape_2_timesteps_feature(x_test,time_steps, features))

loss = model.evaluate(reshape_2_timesteps_feature(x_test,time_steps, features), y_test)

# model.save("./model_cpa/cpa.h5")








# In[262]:


# train: cpa noise => cpa_rate
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp

def reshape_2_timesteps_feature(data_2d, time_steps, features):
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d


    
cpa_noise = np.load("./data/cpa_raw_noise_rnd.npy")
raw_cpa = np.load("./data/cpa_raw_rnd.npy")
row_rate,col_rate = np.gradient(raw_cpa)

cpa_rate = np.log10(np.abs(col_rate))+ 0.5


epoches = 100
time_steps = 64
features = int(raw_cpa.shape[1]/time_steps)

# x_train, x_test, y_train, y_test  = train_test_split(cpa_noise, raw_cpa, test_size=0.2)

# cpa_noise => cpa_rate 
x_train, x_test,   y_train_rate, y_test_rate  = train_test_split(cpa_noise, cpa_rate, test_size=0.2)

display(x_train.shape)


model = tf.keras.Sequential(
[

    tf.keras.layers.LSTM(32, return_sequences=True, input_shape= (time_steps, features) ), 
    tf.keras.layers.Dropout(0.22),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(y_train.shape[1])

]

)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])


history = model.fit( reshape_2_timesteps_feature(x_train,time_steps, features), 
                    y_train_rate, epochs=epoches, batch_size=512 , callbacks=[early_stop] )

# y_pred = model.predict(reshape_2_timesteps_feature(x_test,time_steps, features))

loss = model.evaluate(reshape_2_timesteps_feature(x_test,time_steps, features), y_test_rate)

# model.save("./model_cpa/cpa_rate.h5")

print("- model save ok")


# In[53]:


# plot train history info: cpa noise train to rate 
from math import atan2, pi, cos,sin

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
import pandas as pd 


x=np.int32(np.linspace(0,512,512))
df = pd.DataFrame(history.history)

# plt.plot(df['loss'].loc[2:]/10, label="loss")
plt.plot(df['mae'].loc[2:], label="mae")
plt.xlabel("迭代周期")
plt.ylabel("误差mae")
plt.legend()


plt.show()



# In[57]:


#  inference: cpa noise => cpa && cpa noise => cpa_rate 

from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from sklearn.linear_model import Lasso


def cub_interp(y,cub_sz):
    x=np.arange(len(y))
    cs=CubicSpline(x,y)
    xi=np.linspace(0,len(x)-1,cub_sz*len(x))
    yi = cs(xi)
    return [xi,yi]

def lasso_model(x,y,alpha=0.2):
    x=x.reshape(-1,1)
#     X=np.column_stack((np.power(x,0), x, np.power(x,2), np.power(x,3), np.power(x,4)))
    X=np.column_stack(( x, np.power(x,2), np.power(x,3)))
    lasso = Lasso(alpha=alpha)
    lasso.fit(X,y)
    y_pred = lasso.predict(X)
    lasso.coef_
    return y_pred


def kalman_filter(y):
#     initial part
    Q=1e-3
    R=0.1**2
    F=np.array([[1,1],[0,1]])
    H =np.array([[1,0]])

    x0 = np.array([0,0])
    P0=np.eye(2)*1000    
#  end initial

    x_pred=x0
    P=P0
    x_flt=[]
    
    for yi in y:
        x_pred = F @ x_pred
        P_pred = F @ P @ F.T + Q
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x_flt.append((H @ x_pred)[0])
        x_pred = x_pred + K @ (yi - H @ x_pred)
        P = (np.eye(2) - K @ H ) @ P_pred
    return np.array(x_flt)



def reshape_2_timesteps_feature(data_2d, time_steps, features):
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d

def show_idx_figure_cpa(idx, arr_in_cpa):
    e_view_theta = arr_in_cpa[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()

    
cpa_noise = np.load("./data/cpa_raw_noise_rnd.npy")
raw_cpa = np.load("./data/cpa_raw_rnd.npy")


print(raw_cpa.shape)
row_rate,col_rate = np.gradient(raw_cpa)

print(col_rate.shape)

cpa_rate = np.log10(np.abs(col_rate))+ 0.5

print(cpa_rate.shape)


epoches = 100
time_steps = 64
features = int(raw_cpa.shape[1]/time_steps)

x_train, x_test, y_train, y_test  = train_test_split(cpa_noise, raw_cpa, test_size=0.2,shuffle=False)
# cpa_noise => cpa_rate 
x_train_f, x_test_f, y_train_rate, y_test_rate  = train_test_split(cpa_noise, cpa_rate, test_size=0.2,shuffle=False)




model_cpa = tf.keras.models.load_model("./model_cpa/cpa.h5")
model_cpa.summary()

model_cpa_rate = tf.keras.models.load_model("./model_cpa/cpa_rate.h5")
model_cpa_rate.summary()


y_test_ = model_cpa.predict(reshape_2_timesteps_feature(x_test[0:1200], time_steps, features))
y_test_rate_ = model_cpa_rate.predict(reshape_2_timesteps_feature(x_test[0:1200], time_steps, features))







colors = ['r', 'g', 'b']
markers = ['.', '.', 's']

x = np.linspace(0,512,512)
x = np.int32(x)


row_g, col_g = np.gradient(y_test_)
y_test_cal_rate  = np.log10(np.abs(col_g+0.000001))+ 0.5


for idx in [11,112,114]:
#     idx=9

    plt.scatter(y_test_[idx], x, c=colors[0], marker=markers[0], s=1, label="预测值")
    plt.scatter(y_test[idx],  x, c=colors[1], marker=markers[0], s=1, label="理想真值")
    plt.scatter(x_test[idx],  x, c=colors[2], marker=markers[0], s=1,label="仿真观测值")
    plt.xlabel("角度")
    plt.ylabel("时间")
    plt.legend()
    plt.show()

    print(idx)
    plt.scatter(x_test[idx],  x, c=colors[1], marker=markers[0], s=1)
    plt.xlabel("角度")
    plt.ylabel("时间")
    plt.legend()
    plt.show()
    
    
    g=np.gradient(y_test[idx])
    y_test_rate_orig  = np.log10(np.abs(g)+1e-6)+ 0.5
    
    idx_max = np.argmax(y_test_rate[idx])
    plt.scatter(x[:idx_max], y_test_rate_[idx,:idx_max], c='black', marker=markers[0], s=1,label="预测值_前")
    plt.scatter(x[idx_max:], y_test_rate_[idx,idx_max:], c='red', marker=markers[0], s=1,label="预测值_后", alpha=0.6)    
    plt.scatter(x,  y_test_cal_rate[idx],  c=colors[2], marker=markers[1],s=1, alpha=0.3, label="滤波前值")
    
#     plt.scatter(x, y_test_rate_orig, c='b', marker=markers[0], s=2, label="理想真值")
    
    plt.xlabel("时间")
    plt.ylabel("方位角变化率半对数")
    plt.legend()

    plt.show()
    

    
#     plt.scatter(x, y_test_rate_[idx], c=colors[0], marker=markers[0], s=1,label="预测值")
    plt.scatter(x,  y_test_rate[idx],  c=colors[1], marker=markers[1],s=1,label="理想真值")
    plt.scatter(x,  y_test_cal_rate[idx],  c=colors[2], marker=markers[1],s=1)
    plt.ylabel("方位角变化率半对数")
    plt.xlabel("时间")    
    plt.legend()
    plt.show()

    
    plt.scatter(y_test_[idx, :idx_max-11], y_test_rate_[idx,:idx_max-11], c='black', marker=markers[0], s=1,label="预测值_前")
    plt.scatter(y_test[idx, idx_max-11:], y_test_rate_[idx,idx_max-11:], c='red', marker=markers[0], s=1,label="预测值_后", alpha=0.4)       
#     plt.scatter(y_test[idx],y_test_rate[idx],  c=colors[0], marker=markers[1],s=1,label="理想真值")    
#     plt.scatter(y_test_[idx],y_test_rate_[idx] , c=colors[1], marker=markers[1],s=1,label="预测值")    
    
    plt.ylabel("方位角变化率半对数")
    plt.xlabel("方位角")   
    plt.legend()
    plt.show()    
    

#     y_r_g = np.gradient(y_test_rate_[idx])
#     xc,y_test_rate_cub_=cub_interp(y_test_rate_[idx],7)
    y_test_rate_lasso_ = lasso_model(y_test[idx],y_test_rate_[idx],alpha=0.1)
    
    g_y_test_rate_ = np.abs(np.gradient(y_test_rate_lasso_))
    g_y_test_rate_raw_ = np.abs(np.gradient(y_test_rate_[idx]))
    
#     y_r_g_kal=kalman_filter(y_r_g)
    
#     [x_yrg, y_r_g_cub]= cub_interp(y_r_g, 7)

    plt.scatter(y_test[idx],g_y_test_rate_raw_,  c=colors[0], marker=markers[1],s=1,label="方位变化率的变化率_未滤波")
    plt.scatter(y_test[idx],g_y_test_rate_,  c=colors[1], marker=markers[1],s=1,label="方位变化率的变化率_滤波后")


#     plt.scatter(y_test[idx],y_r_g,  c=colors[0], marker=markers[1],s=1,label="方位变化率的变化率") 
#     plt.scatter(y_test[idx],y_r_g_kal,  c=colors[1], marker=markers[1],s=1,label="方位变化率的变化率") 
    plt.ylabel("方位角变化率半对数的变化率")
    plt.xlabel("方位角")  
    plt.legend()
    plt.show() 
    
    
    
    


    

loss_cpa = model_cpa.evaluate(reshape_2_timesteps_feature(x_test,time_steps, features), y_test)
# loss_cpa_rate = model_cpa_rate.evaluate(reshape_2_timesteps_feature(x_test,time_steps, features), y_test_rate)



# In[286]:


#  inference: cpa noise => cpa
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from tensorflow.keras.utils import plot_model


def reshape_2_timesteps_feature(data_2d, time_steps, features):
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d

def show_idx_figure_cpa(idx, arr_in_cpa):
    e_view_theta = arr_in_cpa[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()

    
time_steps = 64
features = int(raw_cpa.shape[1]/time_steps)    
    
# cpa_real = pd.read_csv("./data/bearing.txt", sep=" ", header=None)


np_bearing =  np.loadtxt("./data/bearing.txt")

np_b = np.abs(np_bearing-360)
np_b

sz=[np_b.shape[0]//512,512]
pad_num = (sz[0] + 1) *  sz[1] - np_b.shape[0]

np_b_pad = np.pad(np_b, (1,pad_num-1), 'symmetric', reflect_type='odd').reshape(512,-1)


print(np_b_pad)
np_b_pad_median = np.median(np_b_pad, axis=1)

np_b_pad_median

x = np.arange(0,512,1)

plt.scatter(np_b_pad_median,x,s=1)
plt.show()

model_cpa = tf.keras.models.load_model("./model_cpa/cpa.h5")
model_cpa.summary()
plot_model(m_0,  to_file='m_0.png',show_shapes=True)


t= np.zeros([1,512]) 
t[0, :] = np_b_pad_median


np_b_pad_median_ = model_cpa.predict(reshape_2_timesteps_feature(t, time_steps, features))


plt.scatter(x, np_b_pad_median_,s=1,c='r')
plt.scatter(x, np_b_pad_median, s=1,c='g')

plt.show()




# In[385]:


def np_move_avg(a,n , mode="same"):
    a = np.pad(a, (32,32), 'edge')
#     a = np.pad(a, (32,32), 'mean') 
    a = np.convolve(a,np.ones((n,))/n,mode=mode)
    a_len = len(a)
    return a[32:a_len-32]

def gen_x(y):
    x=range(len(y))
    return x
x = gen_x(np_b_pad_median)
# plt.scatter(x, np_b_pad_median_,s=1,c='r')
plt.scatter(x, np_b_pad_median, s=1,c='g')

np_b_pad_median_new = np_move_avg(np_b_pad_median, 32)

np_b_pad_median_new.shape


plt.scatter(np.arange(len(np_b_pad_median_new)), np_b_pad_median_new, s=1,c='r') 
plt.scatter(x, np_b_pad_median_new, s=1,c='g') 
np_b_pad_median_new = np_move_avg(np_b_pad_median_new, 16)
plt.scatter(x, np_b_pad_median_new, s=1,c='b') 


plt.show()

np_b_pad_median_new.shape
plt.plot(np_b_pad_median_new - np_b_pad_median)
plt.show()

noise_sim = np_b_pad_median_new - np_b_pad_median

# np.save("./data/noise_sim.npy", noise_sim) 













# In[573]:


# _*_ coding: utf-8 _*_
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.interpolate import CubicSpline
from sklearn.linear_model import Lasso



font_path = "./font/STFANGSO.TTF"
font_name = FontProperties(fname=font_path).get_name()

plt.rcParams['font.family'] = [font_name] 

def exp_avg(y, alpha):
#     alpha=0.6
    s=np.zeros_like(y)
    s[0]=y[0]
    for i in range(1,len(y)):
        s[i]=alpha*y[i]+(1-alpha)*s[i-1]    
    return s

def gen_x(y):
    x=range(len(y))
    return x


def get_noise(noise,len):
    s_idx = np.random.permutation(len)
    return noise[s_idx]

def np_move_avg(a,n , mode="same"):
    pad_num = 32
    a = np.pad(a, (pad_num,pad_num), 'edge')
#     a = np.pad(a, (32,32), 'mean') 
    a = np.convolve(a,np.ones((n,))/n,mode=mode)
    a_len = len(a)
    return a[pad_num:a_len-pad_num]

def cub_interp(y,cub_sz):
    x=np.arange(len(y))
    cs=CubicSpline(x,y)
    xi=np.linspace(0,len(x)-1,cub_sz*len(x))
    yi = cs(xi)
    return [xi,yi]

def win_smooth(y, window_size=5):
    if not (window_size % 2):
        window_size += 1
    window_half_size = (window_size-1)//2
    weights = np.arange(1,window_size+1).astype(np.float32)
    
    weights /= weights.sum()
    y_smooth = np.convolve(y, weights, mode='valid')
    
    return np.concatenate([y_smooth[0]*np.ones(window_half_size),  y_smooth, y_smooth[-1]*np.ones(window_half_size)])


from scipy import optimize
def fit_func(p,x):
    m,a,b,c=p
    return m*x**3 - a*x**2 + b*x + c

def err_func(p,x,y):
    return fit_func(p,x)-y

def my_leastsq(err_func, x,y):
    p0=[0.01,-0.02,0.03,0]
    params, success = optimize.leastsq(err_func, p0, args=(x,y))
    print("- success: ", success)
    y_smooth = fit_func(params,x)
    return y_smooth

def kalman_filter(y):
#     initial part
    Q=1e-5
    R=0.1**2
    F=np.array([[1,1],[0,1]])
    H =np.array([[1,0]])

    x0 = np.array([0,0])
    P0=np.eye(2)*1000    
#  end initial

    x_pred=x0
    P=P0
    x_flt=[]
    
    for yi in y:
        x_pred = F @ x_pred
        P_pred = F @ P @ F.T + Q
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x_flt.append((H @ x_pred)[0])
        x_pred = x_pred + K @ (yi - H @ x_pred)
        P = (np.eye(2) - K @ H ) @ P_pred
    return np.array(x_flt)


def lasso_model(x,y,alpha=0.2):
    x=x.reshape(-1,1)
#     X=np.column_stack((np.power(x,0), x, np.power(x,2), np.power(x,3), np.power(x,4)))
    X=np.column_stack(( x, np.power(x,2), np.power(x,3)))
    lasso = Lasso(alpha=alpha)
    lasso.fit(X,y)
    y_pred = lasso.predict(X)
    lasso.coef_
    return y_pred


deg = np.load("./data/sim_little_degree.npy")
d=deg
x_=np.arange(len(d)).astype(np.float32)
display(type(x_))

x = np.zeros_like(deg)

for i in range(len(x)):
    x[i] = x_[i]




noise_sim = np.load("./data/noise_sim.npy")



noise = get_noise(noise_sim, 512)




dn=deg+noise




ds_kal = kalman_filter(dn)



plt.plot(dn, x, label='测量值', alpha=0.3)
plt.plot(d,x, label='理想真值', c='g')
plt.plot(ds_kal[3:], x[3:], label='kalman滤波器', c='y', alpha=1)  

# plt.plot(ds_06, label='拟合线_alpha_0.6', c='yellow')
# plt.plot(ds_09, label='拟合线_alpha_0.8',c='yellow')



plt.xlabel("方位角")
plt.ylabel("时间")

plt.legend()
plt.show()





# In[40]:


#  inference: cpa noise => cpa && cpa noise => cpa_rate 

from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
def gen_x(y):
    x=range(len(y))
    return x

def reshape_2_timesteps_feature(data_2d, time_steps, features):
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d

def show_idx_figure_cpa(idx, arr_in_cpa):
    e_view_theta = arr_in_cpa[idx]
    step_arr = np.zeros(e_view_theta.shape)

    step_arr= range(0,step_arr.shape[0]) 

    plt.scatter(e_view_theta, step_arr)
    plt.show()

    
cpa_noise = np.load("./data/cpa_raw_noise_rnd.npy")
raw_cpa = np.load("./data/cpa_raw_rnd.npy")


print(raw_cpa.shape)
row_rate,col_rate = np.gradient(raw_cpa)

print(col_rate.shape)

cpa_rate = np.log10(np.abs(col_rate))+ 0.5

print(cpa_rate.shape)


epoches = 100
time_steps = 64
features = int(raw_cpa.shape[1]/time_steps)

x_train, x_test, y_train, y_test  = train_test_split(cpa_noise, raw_cpa, test_size=0.2,shuffle=False)
# cpa_noise => cpa_rate 
x_train_f, x_test_f, y_train_rate, y_test_rate  = train_test_split(cpa_noise, cpa_rate, test_size=0.2,shuffle=False)

idx=1101
d=y_test[idx]

plt.scatter(d, gen_x(d), s=1, label="真值：方位角/方位")
plt.xlabel("方位角")
plt.ylabel("方位")
plt.legend()


plt.show()


c_d = np.gradient(d)
c_d = np.log10(np.abs(c_d+1e-6))+0.5
plt.scatter(d,c_d,s=1)


plt.xlabel("方位角")
plt.ylabel("方位角变化率半对数")
plt.show()






# In[43]:


from matplotlib.font_manager import FontProperties
from scipy.interpolate import CubicSpline
from scipy import optimize
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures


#  inference: cpa noise => cpa
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from tensorflow.keras.utils import plot_model


font_path = "./font/STFANGSO.TTF"
font_name = FontProperties(fname=font_path).get_name()

plt.rcParams['font.family'] = [font_name] 



model_cpa_test =tf.keras.models.load_model("./model_cpa/cpa.h5")

model_cpa_test.summary()


plot_model(model_cpa_test,  to_file='m_0.png',show_shapes=True)























# In[98]:


# speed change in each course 

from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp



def get_angle_degrees(v1, v2):
    # 计算两个向量的点积
    dot_product = np.dot(v1, v2)

    # 计算两个向量的模长
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    # 计算两个向量的夹角（弧度）
    angle_in_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

    # 将弧度转换为角度
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees


def to_unit(v):
    # 定义向量v
    v 
    # 计算向量v的模长
    magnitude = np.linalg.norm(v)

    # 计算向量v的单位向量
    unit_vector = v / magnitude

    return unit_vector



# initial : 
# uav_loc = [0,0]

# R
# view_theta_r0
# => obj_loc = [R*sin(view_theta_r0), R*cos(view_theta_r0)]
 
# v_obj = [1,-2]   => v_obj_unit 

# speed_obj = 0.2

# for step_cnt in range(2000):
#     obj_loc + v_obj_unit * speed_obj
#     print (obj_loc) 
# view_theta_new = acos( []

### global var ###
global uav_loc
global R
global view_theta_r0
global v_obj
global speed_obj
global max_range

global arr_out

uav_loc = np.array([0,0])
R=100
v_obj = np.array([1,-3])
speed_obj = 1
view_theta_r0 = 3
max_range = 512

##################


def gen_array_with_args(idx, uav_loc, R, v_obj, speed_obj, view_theta_r0, max_range):
    global arr_out
    
    arr_out[idx,0]=idx
    arr_out[idx,1:1+2]=uav_loc
    arr_out[idx,3] =R
    arr_out[idx,4:4+2] = v_obj
    arr_out[idx,6]= speed_obj
    arr_out[idx,7]= view_theta_r0
    
    
    v_obj=to_unit(v_obj)
    view_theta_r0_rad = np.deg2rad(view_theta_r0)
    obj_loc = np.array([R*sin(view_theta_r0_rad), R*cos(view_theta_r0_rad)])

    

    obj_loc_set = np.zeros((max_range,2))
    view_theta_degree_set = np.zeros((max_range,2))
    
    for step_cnt in range(max_range):
        if step_cnt % 22 == np.random.randint(22):
            speed_obj = speed_obj + (np.random.randint(10)-5)*0.0003
            
        obj_loc = obj_loc + v_obj * speed_obj
        obj_loc_set[step_cnt,:] = obj_loc
        
        view_theta_degree = get_angle_degrees([0,1], obj_loc-uav_loc)
        view_theta_degree_set[step_cnt,:] = [view_theta_degree,step_cnt]

#     disp(obj_loc_set)

    
#     plt.scatter(obj_loc_set[:,0], obj_loc_set[:,1])
#     plt.scatter(view_theta_degree_set[:,0], view_theta_degree_set[:,1])
    
    arr_out[idx,8:8+512] = view_theta_degree_set[:,0]
    
    
    
#     plt.show()
 



#main_ 
def main():
    global uav_loc #观察位置
    global R   #半径
    global view_theta_r0  #角度
    global v_obj  #
    global speed_obj #速度
    global max_range  #观察次数
    
    v_obj=to_unit(v_obj)
    view_theta_r0_rad = np.deg2rad(view_theta_r0)
    obj_loc = np.array([R*sin(view_theta_r0_rad), R*cos(view_theta_r0_rad)])

    

    obj_loc_set = np.zeros((max_range,2))
    view_theta_degree_set = np.zeros((max_range,2))
    
    for step_cnt in range(max_range):
        obj_loc = obj_loc + v_obj * speed_obj
        obj_loc_set[step_cnt,:] = obj_loc
        
        view_theta_degree = get_angle_degrees([0,1], obj_loc-uav_loc)
        view_theta_degree_set[step_cnt,:] = [view_theta_degree,step_cnt]

#     disp(obj_loc_set)

    
#     plt.scatter(obj_loc_set[:,0], obj_loc_set[:,1])
    plt.scatter(view_theta_degree_set[:,0], view_theta_degree_set[:,1])
    
    plt.show()


# uav_loc = 

arr_out= np.zeros([50000,8+512], dtype=np.float32)



print(uav_loc.shape[0])



# main()
np.random.seed(0)
speed_set = np.random.random(10)*0.004 +0.0094
idx = 0

for v_obj_ in np.array([[1,-3], [1,-3.3], [1,-4], [1,-6], [1.3,-5.1]]):
    for speed_obj_ in speed_set:
        speed_obj = speed_obj_
        
        
        for R_ in np.linspace(150,500,20):
            R = R_
            for theta_ in np.linspace(1,77,50):
                view_theta_r0 = theta_
                if(idx%1000==0):
                    print("- idx is ", idx)
                
                gen_array_withA_args(idx, uav_loc, R, v_obj, speed_obj, view_theta_r0, max_range)
                idx = idx+1
                
# arr_out[0,:]
np.save("./data/cpa_.npy", arr_out)
print("- save cpa ok")



# In[102]:


cpa=np.load("./data/cpa_raw_.npy")

x=np.arange(0,cpa.shape[1])

plt.scatter(cpa[np.random.randint(999)],x, s=1) 
plt.show()


# In[129]:


noise_sim = np.load("./data/noise_sim.npy")

plt.plot(noise_sim)
plt.show()

cpa=np.load("./data/cpa_raw_.npy")


np.random.shuffle(cpa)
# np.save("./data/cpa_raw_.npy", cpa)

print("- save cpa shuffle ")

def get_noise(noise,len_):
    s_idx = np.random.permutation(len_)
    return noise[s_idx]



for i in np.arange(cpa.shape[0]):
    if i % 10000 == 0:
        print(i)
    
    cpa[i] = cpa[i] + get_noise(noise_sim, len(cpa[0]))

plt.scatter(cpa[0] , x,s=1)
plt.show()



# np.save("./data/cpa_noise_.npy", cpa)

print("- save cpa noise ok")

    


# In[138]:


cpa_n = np.load("./data/cpa_noise_.npy")
cpa = np.load("./data/cpa_raw_.npy")

x=np.arange(len(cpa[0]))
rnd = np.random.randint(len(x))
plt.scatter(cpa[rnd], x , s=1)
plt.scatter(cpa_n[rnd], x, s=1)

plt.show()




# In[193]:


c_g = np.gradient(cpa[0])
x=np.arange(len(cpa[0]))
rnd = np.random.randint(len(x))

idx = rnd
plt.plot(cpa[idx])
plt.show()

r_g, c_g = np.gradient(cpa)



cpa_rate = np.log10(np.abs(c_g) + 1e-7)+ 0.5
plt.plot(cpa_rate[idx])
plt.show()


# In[339]:


# train both 

import tensorflow as tf
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties



def reshape_2_timesteps_feature(data_2d, time_steps, features):
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d

def use_font():
    font_path = "./font/STFANGSO.TTF"
    font_name = FontProperties(fname=font_path).get_name()

    plt.rcParams['font.family'] = [font_name] 

use_font()








x_train, x_test, y_train, y_test  = train_test_split(cpa_n, cpa, test_size=0.2,shuffle=False)

x_train_r, x_test_r, y_train_rate, y_test_rate  = train_test_split(cpa_n, cpa_rate, test_size=0.2,shuffle=False)


assert(np.all(x_train[0] == x_train_r[0]))

time_steps = 32
batch_size=512
features = int(cpa.shape[1]/time_steps)
epoches = 1000
_L=tf.keras.layers

model_cpa = tf.keras.Sequential(
[
    _L.Dense(units=32, activation="relu", input_shape=x_train.shape[1:]),
#     _L.Dropout(0.3),
    _L.Dense(units=16, activation="relu"),
       
    _L.Dense(units=y_train.shape[1])
]

)


model_cpa_rate = tf.keras.Sequential(
[

    tf.keras.layers.LSTM(32, return_sequences=True, input_shape= (time_steps, features) ), 
    tf.keras.layers.Dense(y_train.shape[-1]//4 , input_shape=x_train.shape[1:]),
    tf.keras.layers.Dropout(0.22),
    tf.keras.layers.Dense(y_train.shape[-1]//8),
    tf.keras.layers.Dropout(0.32),

    tf.keras.layers.Dense(y_train.shape[1])

]

)


# tf.keras.optimizers.schedules.le
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=3)

model_cpa.compile(loss='mae', optimizer=tf.optimizers.Adam(learning_rate=1e-4), metrics=['mae'])

model_cpa.summary()

# history_cpa = model_cpa.fit(x_train, y_train, epochs=epoches, batch_size=batch_size , callbacks=[early_stop], validation_data=(x_test,y_test))
history_cpa = model_cpa.fit(x_train, y_train, epochs=epoches, batch_size=512 , 
                            callbacks=[early_stop], validation_data=(x_test,y_test))
model_cpa.save("./model_cpa/cpa.h5")
print("- model cpa save ok ")
loss_cpa = model_cpa.evaluate(x_test, y_test)


# model_cpa_rate.compile(loss='mse', optimizer='adam', metrics=['mae'])
# history_cpa_rate = model_cpa_rate.fit(reshape_2_timesteps_feature(x_train_r,time_steps, features), y_train_rate, epochs=epoches, batch_size=512 , callbacks=[early_stop])
# model_cpa_rate.save("./model_cpa/cpa_rate.h5")
# print("- model cpa rate save ok ")






# In[364]:


# retrain again 
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=4)
model_cpa =tf.keras.models.load_model("./model_cpa/cpa-1048.h5")

model_cpa.compile(loss='mae', optimizer=tf.optimizers.Adam(learning_rate=1e-10), metrics=['mae'])
history_cpa = model_cpa.fit(x_train, y_train, epochs=epoches, batch_size=64 ,  shuffle=True,                          
                            callbacks=[early_stop], validation_data=(x_test,y_test))

model_cpa.save("./model_cpa/cpa.h5")
print("- model cpa save ok ")
loss_cpa = model_cpa.evaluate(x_test, y_test)





# In[518]:


# re-train , fine-tune 

import tensorflow as tf
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties

import copy

global noise_idx 
noise_idx = 0

def get_noise(noise,len):
    global noise_idx
    s_idx = np.random.permutation(len)
    
    len_ = len
    noise_factor = np.random.randint(low=0, high=10, size=len_)/10 + (np.random.random(len_)-0.5)/10     
    
    if (noise_idx % 3 == np.random.randint(3)):
        noise[s_idx] = np.multiply( noise[s_idx], noise_factor)
    noise_idx += 1
    return noise[s_idx]





_L=tf.keras.layers

noise_sim = np.load("./data/noise_sim.npy")






cpa = np.load("./data/cpa_raw_.npy")

batch_size=512
time_steps = 32
features = int(cpa.shape[1]/time_steps)
epoches = 25
start = 60


print( "- start re-train ")

while True:
    cpa = np.load("./data/cpa_raw_.npy")
    cpa_n = copy.deepcopy(cpa)
#     print("- a course")
    i=0
    for e in cpa:
        if (i % 3  == np.random.randint(3)):
            cpa[i] = cpa[i][::1]
        cpa_n[i] = cpa[i] + get_noise(noise_sim, len(cpa[0]))
        rnd = np.random.randint(start, len(cpa[0]))
        cpa_n[i,rnd:] = 0
        cpa[i,rnd:] = 0
        
        i+=1
    
#     print("- done deal cpa cpa_n")
 

    x_train, x_test, y_train, y_test  = train_test_split(cpa_n, cpa, test_size=0.2,shuffle=False)

    # retrain again 
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=5)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath = "./model_cpa/cpa.h5",
        save_best_only=True,
        monitor='val_mae',
        mode="min",
        verbose=0
    )
    model_cpa =tf.keras.models.load_model("./model_cpa/cpa.h5")
    
    loss_cpa = model_cpa.evaluate(x_test, y_test)
    
    
    model_cpa.compile(loss='mae', optimizer=tf.optimizers.Adam(learning_rate=1e-4), metrics=['mae'])
    
    history_cpa = model_cpa.fit(x_train, y_train, epochs=epoches, batch_size=512 ,  shuffle=True,                          
                                callbacks=[early_stop,checkpoint_cb], validation_data=(x_test,y_test),
                                verbose=0
                               )

#     model_cpa.save("./model_cpa/cpa.h5")
#     print("- model cpa save ok ")





# In[519]:


# inference from cpa_n => cpa , cpa_rate
import pandas as pd 
from sklearn.linear_model import Lasso

def lasso_model(y,alpha=0.2):
    pad_num = 55
    y_len_old = len(y)
    
    y=np.pad(y, (pad_num, pad_num), 'symmetric', reflect_type='odd' )
    x = np.arange(len(y))
    
    x=x.reshape(-1,1)
#     X=np.column_stack((np.power(x,0), x, np.power(x,2), np.power(x,3), np.power(x,4)))
    X=np.column_stack(( x, np.power(x,2), np.power(x,3)))
    lasso = Lasso(alpha=alpha)
    lasso.fit(X,y)
    y_pred = lasso.predict(X)
    lasso.coef_
    return y_pred[pad_num:y_len_old+pad_num]

def get_noise(noise,len):
    s_idx = np.random.permutation(len)
    return noise[s_idx]


model_cpa =tf.keras.models.load_model("./model_cpa/cpa.h5")
model_cpa.summary()
noise_sim = np.load("./data/noise_sim.npy")
cpa = np.load("./data/cpa_raw_.npy")
cpa_n = np.load("./data/cpa_noise_.npy")


plt.plot(cpa[177])
plt.plot(cpa_n[177])
plt.show()


x_train, x_test, y_train, y_test  = train_test_split(cpa_n, cpa, test_size=0.2,shuffle=False)

# loss_cpa = model_cpa.evaluate(reshape_2_timesteps_feature(x_test,time_steps, features), y_test)
start = 80

x_test[:,start:] = 0
y_test[:,start:] = 0




loss_cpa = model_cpa.evaluate(x_test, y_test)
y_test_ = model_cpa.predict(x_test)

for idx in [1777,222,333,123]:

    plt.plot(x_test[idx,:start], c='yellow')
    plt.plot(y_test[idx,:start],c='b')
    plt.plot(y_test_[idx,:start], c='r')
    plt.show()



assert(0)


# model_cpa_rate =tf.keras.models.load_model("./model_cpa/cpa_rate.h5")
# loss_cpa_rate = model_cpa_rate.evaluate(reshape_2_timesteps_feature(x_test,time_steps, features), y_test_rate)


need = 2000
# y_test_ = model_cpa.predict(reshape_2_timesteps_feature(x_test[0:need], time_steps, features))
y_test_ = model_cpa.predict(x_test[0:need])


max_mae = np.max(np.abs(y_test_-y_test[0:need]), axis=1)


# y_test_rate_ = model_cpa_rate.predict(reshape_2_timesteps_feature(x_test[0:need], time_steps, features))


arr_idx = [np.random.randint(need) for i in range(1)]

x = np.arange(len(x_test[0]))

arr_idx.append(1777)


for i in arr_idx:
#     break
    print(i)
    plt.plot(y_test_[i],x, c='r')
    plt.plot(y_test[i],x, c='g')
    plt.plot(x_test[i],x, alpha=0.7)
    plt.plot(lasso_model(y_test_[i]),x, c='black')
#     plt.plot(lasso_model(x_test[i]),x, c='purple')
    plt.show()
    
#     plt.plot(y_test_rate_[i], c='r')
#     plt.plot(y_test_rate[i])
#     plt.show()


# plt.plot(max_mae)
    




for i in arr_idx:    
    break
    plt.plot(x_train[i])
    plt.plot(y_train[i])
    plt.show()
    
    plt.plot(x_test[i])
    plt.plot(y_test[i])
    plt.show()


# In[523]:


from sklearn.linear_model import Lasso

def lasso_model(x,y,alpha=0.22):
    x=x.reshape(-1,1)
#     X=np.column_stack((np.power(x,0), x, np.power(x,2), np.power(x,3), np.power(x,4)))
    X=np.column_stack(( x, np.power(x,2), np.power(x,3)))
    lasso = Lasso(alpha=alpha)
    lasso.fit(X,y)
    y_pred = lasso.predict(X)
    lasso.coef_
    return y_pred



model_cpa =tf.keras.models.load_model("./model_cpa/cpa.h5")

cpa_real = np.loadtxt("./data/bearing.txt")
start = 222

idx=1
cpa_real = cpa_real[512*idx:512*(idx+1)]


cpa_real[start:] = 0



cpa_real =  cpa_real.reshape(-1,len(cpa_real))
cpa_real_ = model_cpa.predict(cpa_real)

cpa_real = cpa_real[0]
cpa_real_ = cpa_real_[0]


P(cpa_real[:start])
P(cpa_real_[:start], c='r')
PS()







assert(0)



# cpa_real = (cpa_real - 360)*(-1)
idx=555

cpa_real = cpa_real 

cpa_real_p = cpa_real[idx:idx+512]

disp(cpa_real_p.shape)

cpa_real_pt= np.zeros([1,512]) 
cpa_real_pt[0, :] = cpa_real_p

cpa_real_p = cpa_real_pt

cpa_real_p_  = model_cpa.predict(cpa_real_p)

x = np.arange(len(cpa_real_p[0]))
# plt.plot(np.abs(cpa_real_p[0]-360), x)
# plt.plot(np.abs(cpa_real_p_[0]-360),x)


plt.plot(cpa_real_p[0], x)
plt.plot(cpa_real_p_[0],x)

# cpa_real_p_lasso = lasso_model(x,np.abs(cpa_real_p_[0]-360))
cpa_real_p_lasso = lasso_model(x,cpa_real_p_[0])
plt.plot(cpa_real_p_lasso,x, c='black')

plt.show()







# In[161]:


cpa_real = np.loadtxt("./data/bearing.txt")

disp(cpa_real.shape)





cpa_real = 360-cpa_real

t=np.zeros([4,512])
# t[0,:] = cpa_real[0:512]
idx=2
for idx in [0,1,2]:
    t[idx,:] = cpa_real[512*idx:512*(idx+1)]
idx=3  

r_pad_num = 512- len(cpa_real) % 512
r_pad_num = r_pad_num % 512

cpa_real_left = np.pad(cpa_real[512*idx:], (0,r_pad_num), 'symmetric', reflect_type='odd' )

print(cpa_real_left.shape)

t[idx] = cpa_real_left





t_=model_cpa.predict(t)

t_all = np.concatenate([t_[0],t_[1],t_[2], t_[3]])
t_all_lasso = np.concatenate([lasso_model(t_[0]),lasso_model(t_[1]),lasso_model(t_[2]),lasso_model(t_[3])])


x = np.arange(len(t[0])) 
t[0]=360-t[0]
t_[0] = 360-t_[0]
t_lasso = lasso_model(t_[0])
plt.plot(t_[0], x, c='yellow')
plt.plot(t[0],x, alpha=0.5)
plt.plot(t_lasso,x, c='black')

plt.show()

x_tall = np.arange(len(t_all))

plt.plot(t_all, c='r')
plt.plot(cpa_real, alpha=0.5)

plt.plot(t_all_lasso, alpha=0.5, c='purple')

plt.show()



# In[166]:


import copy 
from scipy.interpolate import interp1d



t=np.random.random(3)
print(t)
right_pad_num = 3
np.pad(t,(0,3), 'symmetric', reflect_type='odd')


len_ = 512
noise_factor = np.random.randint(low=0, high=10, size=len_)/10 + (np.random.random(len_)-0.5)/10 



t0= np.linspace(3,1,4)
t1 = np.linspace(1,3,4)
disp(t0)
disp(t1)
np.multiply(t0,t1)

def get_noise(noise,len):
    s_idx = np.random.permutation(len)
    return noise[s_idx]

cpa = np.load("./data/cpa_raw_.npy")
cpa_n = np.load("./data/cpa_noise_.npy")



cpa.shape
cpa_n.shape

t0=np.random.random([3,4])
t1=np.random.random([3,4])

disp(t0)
disp(t1)
d=np.abs(t1-t0)

disp(d)

np.max(d, axis=1)
np.argmax(d, axis=1)

plt.plot(y_test[1])

plt.plot(y_test_[1])
plt.show()


d= np.abs(y_test[0:need] - y_test_)

c_idx = np.argmax(d, axis=1) 

max_ = np.max(d)
r_idx = 0
row = len(d)
col = len(d[0])

for r in range(row):
    for c in range(col):
        
        if (max_ == d[r,c]):
            print(r, c) 




plt.plot(cpa_real)

cpa_real_less = cpa_real[::3]
target_length=512

interp_fn = interp1d(np.arange(0, target_length), cpa_real_less, kind='linear')

upsampled_data = interp_fn(np.linspace(0, target_length-1, original_length))


plt.plot(cpa_real_less)
plt.plot(upsampled_data)
plt.show()






# In[205]:


cpa_real.shape

cpa_real = 360-cpa_real

target_len = 512
scale_factor = cpa_real.shape[0]*1.0 /target_len

x_big = np.arange(len(cpa_real))
x_big.astype(np.float32)

x_big_sc = x_big/scale_factor 
x_big_sc
down_arr = copy.deepcopy(cpa_real[0:target_len])

for i in range(target_len):
    down_arr[i] = cpa_real[int(i*scale_factor)]

    
    
def linear_interp(y, len_y_):
    # from len(y) => len_y_ 
    interp_fn = interp1d(np.arange(len(y)).astype(np.float32), y, kind='linear')
    y_ = interp_fn(np.linspace(0, len(y)-1, len_y_))
    return y_


t=np.linspace(0,9,555)
y = np.power(t,2)

y_ = linear_interp(y, len(y)//2)

plt.plot(y)
plt.show()
plt.plot(y_)
plt.show()




down_arr.shape
down_arr = down_arr.reshape(-1,target_len)

down_arr_ = model_cpa.predict([down_arr])
    
plt.plot(down_arr[0])
plt.plot(down_arr_[0]) 
plt.show()

interp_fn = interp1d(np.arange(len(down_arr_[0])).astype(np.float32), down_arr_[0], kind='linear')

upsampled_data = interp_fn(np.linspace(0, target_len-1, cpa_real.shape[0]))
plt.plot(cpa_real)
plt.plot(upsampled_data)
plt.show()




# In[269]:


def get_noise(noise,len):
    s_idx = np.random.permutation(len)
    return noise[s_idx]



P=plt.plot
PS=plt.show
L_I = linear_interp

ai_len = 512
x=np.linspace(-0.5,0.5,ai_len)
y = np.power(x,2)/2 
y = y+25


y_r0 = y[40:40+100]

y = copy.deepcopy(y_r0)
yn = y + get_noise(noise_sim, len(y))


y512 = L_I(y, ai_len)
yn512 = L_I(yn, ai_len)

y512_ = model_cpa.predict(yn512.reshape(-1,len(yn512)))

P(y512_[0], c='r')

P(yn512)
P(y512,c='g' )

PS()



y20=L_I(y512, len(y_r0))
yn20=L_I(yn512, len(y_r0))
y20_=L_I(y512_[0], len(y_r0))

P(y20)
P(yn20)
P(y20_)
PS()


c_g = np.gradient(x_test[0])
P(np.abs(c_g))
PS()
plt.hist(np.abs(c_g), bins=10)
PS()


c_g = np.gradient(cpa_real)
P(np.abs(c_g[0:512]))
PS()


plt.hist(np.abs(c_g[512:512+512]), bins=10)
PS()
assert(0)













# In[308]:


import os
def sleep_p(sleep_time):
    cmd_sleep = "sleep " + str(sleep_time)
    os.popen(cmd_sleep).readlines()

def run_model_with_point(base_arr, model_cpa_, model_input_dim):
    b_s_r0= len(base_arr)
    base_arr_s = L_I(base_arr, model_input_dim)
    base_arr_s_ = model_cpa_.predict(base_arr_s.reshape(-1,model_input_dim))
    base_arr_s_b_ = L_I(base_arr_s_[0], b_s_r0)
    return base_arr_s_b_


        
for i in range(2):
    t=sleep_p(1)
    
    
    print(i)
print("- done")


# In[654]:


def kalman_filter(y):
#     initial part
    Q=1e-3
    R=0.1**2
    F=np.array([[1,1],[0,1]])
    H =np.array([[1,0]])

    x0 = np.array([0,0])
    P0=np.eye(2)*1000    
#  end initial

    x_pred=x0
    P=P0
    x_flt=[]
    
    for yi in y:
        x_pred = F @ x_pred
        P_pred = F @ P @ F.T + Q
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x_flt.append((H @ x_pred)[0])
        x_pred = x_pred + K @ (yi - H @ x_pred)
        P = (np.eye(2) - K @ H ) @ P_pred
    return np.array(x_flt)


def lasso_model(y,alpha=0.1):
    pad_num = len(y)//16
#     pad_num = 55
    y_len_old = len(y)
    
    y=np.pad(y, (pad_num, pad_num), 'symmetric', reflect_type='odd' )
    x = np.arange(len(y))
    
    x=x.reshape(-1,1)
#     X=np.column_stack((np.power(x,0), x, np.power(x,2), np.power(x,3), np.power(x,4)))
    X=np.column_stack(( x, np.power(x,2), np.power(x,3)))
    lasso = Lasso(alpha=alpha)
    lasso.fit(X,y)
    y_pred = lasso.predict(X)
    lasso.coef_
    return y_pred[pad_num:y_len_old+pad_num]


start_ = 122

str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"
model_id =tf.keras.models.load_model(str_path_save_model)

cpa = np.load("./data/cpa_raw_.npy")
cpa_n = np.load("./data/cpa_noise_.npy")

cpa_n[:,start_:] = 0
cpa[:,start_:]=0

y_test_ = model_id.predict(cpa_n)

y_test_[:,start_:] = 0

y_test__ = model_id.predict(y_test_)
y_test___ = model_id.predict(y_test__)



np.random.seed(0)

rnd_list = np.random.randint(40000,50000, 16)
for rnd_ in rnd_list:
    print(rnd_)
#     P(cpa_n[rnd_,:start_])
    P(y_test_[rnd_,:start_],c="r")
    
    y_test_kalman_ = kalman_filter(y_test_[rnd_,:start_])
    
    P(y_test__[rnd_,:start_],c="black")
    
#     P(y_test___[rnd_,:start_],c="r")
    
#     y_test_lasso___=lasso_model(y_test_[rnd_,:start_])
#     P(y_test_lasso___,c="purple")
    
#     y_test_lasso____=lasso_model(y_test___[rnd_,:start_])
#     P(y_test_lasso____,c="black")
#     P(y_test_kalman_,c="yellow")
    
    
    P(cpa[rnd_,:start_], c='g')
    PS()



# In[349]:


# %%time 
cpa_real = np.loadtxt("./data/bearing.txt")

cpa_real = x_test[0]

def win_smooth(y, window_size=5):
    if not (window_size % 2):
        window_size += 1
    window_half_size = (window_size-1)//2
    weights = np.arange(1,window_size+1).astype(np.float32)
    
    weights /= weights.sum()
    y_smooth = np.convolve(y, weights, mode='valid')
    
    return np.concatenate([y_smooth[0]*np.ones(window_half_size),  y_smooth, y_smooth[-1]*np.ones(window_half_size)])



y= y_test[0]




start = 0
intv = 60
fetch_loc = -4

cpa_get = list(cpa_real[start:start+intv])

l_cpa = []
l_cpa_ = []
l_y = []

fetch_loc = -10
model_input_dim = 512
end = intv



while (end < len(cpa_real)-1):

    base_arr=cpa_real[start:end]
    base_arr_s = win_smooth(base_arr,22)

    
    
    
    base_arr_ = run_model_with_point(base_arr, model_cpa, model_input_dim)

    l_cpa_.append(base_arr_[fetch_loc])
    l_cpa. append(base_arr [fetch_loc])
    l_y.append(y[start:end][fetch_loc])
    
    

    
#     P(cpa_get, alpha=0.1, c='b')
    if (end % 100 == 0):
        print(end, end=" ")
        
    if (end<model_input_dim):
        pass
    else:
        start += 1
    end+=1
    
    if (start > 1300):
        break

        
P(l_cpa)
P(l_cpa_,c='r')
P(l_y)
PS()

P(x_test[0])
P(y_test[0])
y_test_ = run_model_with_point(x_test[0], model_cpa, model_input_dim)
P(y_test_)
PS()



# In[348]:


c_g = np.gradient(l_cpa_)
r = np.log10(np.abs(c_g) + 1e-6) + 0.5 
plt.scatter(x)








# In[530]:


l=5
start = 2

str_save_model = ""

for start_ in range(start,l+1):
    t_r0=np.random.random([3,l]) + 27
    str_save_model = "- model_name:" + str(start_)
    for model_use_data_change_num in range(4):
        t=copy.deepcopy(t_r0)
        t[:,start_:] = 0
        disp(t)
        print("-----%s", str_save_model)


# In[531]:


format(9,"04")






# ```
# predict_val = avg_val + rate * time 
#     east_x = avg_x + rate * time 
#     ...y = avg_y + rate * time 
#     
# 
# avg = predict_val + alpha * ( cluster_val - predict_val)
#     avg_x = est_x + alpha * ( cos(clust_brg) - track_east_x )
#     ..._y = est_y + alpha * ( sin(xxx) - track_east_y) 
#    
# rate_r1 = rate_r0 + beta *( cluster_val - predict_val) 
# ```
# 
# 
# 
# 
# 

# In[556]:


# re-train , fine-tune , generate 453 models 

import tensorflow as tf
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties

import copy

global noise_idx 
noise_idx = 0

def get_noise(noise,len):
    global noise_idx
    s_idx = np.random.permutation(len)
    
    len_ = len
    noise_factor = np.random.randint(low=0, high=10, size=len_)/10 + (np.random.random(len_)-0.5)/10     
    
    if (noise_idx % 3 == np.random.randint(3)):
        noise[s_idx] = np.multiply( noise[s_idx], noise_factor)
    noise_idx += 1
    return noise[s_idx]





_L=tf.keras.layers

noise_sim = np.load("./data/noise_sim.npy")






# cpa = np.load("./data/cpa_raw_.npy")

# batch_size=512
# time_steps = 32
# features = int(cpa.shape[1]/time_steps)
# epoches = 25
# start = 60


print( "- start re-train ")


l=512
start = 50
model_change_data_times = 25

str_save_model = ""
epoches = 6
cpa_r0 = np.load("./data/cpa_raw_.npy")
for start_ in range(start,l+1):

    str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"
    print(str_path_save_model)
    
    for _times in range(model_change_data_times):
        cpa=copy.deepcopy(cpa_r0)
        cpa_n = copy.deepcopy(cpa)
    #     print("- a course")
        idx=0
        for e in cpa:
            if (idx % 3  == np.random.randint(3)):
                cpa[idx] = cpa[idx][::1]
            cpa_n[idx] = cpa[idx] + get_noise(noise_sim, len(cpa[0]))
            rnd = np.random.randint(start_, len(cpa[0])+1)
            cpa_n[idx,rnd:] = 0
            cpa[idx,rnd:] = 0

            idx+=1

    #     print("- done deal cpa cpa_n")


        x_train, x_test, y_train, y_test  = train_test_split(cpa_n, cpa, test_size=0.2,shuffle=False)

        # retrain again 
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=2)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath = str_path_save_model,
            save_best_only=True,
            monitor='val_mae',
            mode="min",
            verbose=0
        )
        model_cpa =tf.keras.models.load_model(str_path_save_model)
        loss_cpa = model_cpa.evaluate(x_test, y_test)
        model_cpa.compile(loss='mae', optimizer=tf.optimizers.Adam(learning_rate=1e-4), metrics=['mae'])
        history_cpa = model_cpa.fit(x_train, y_train, epochs=epoches, batch_size=512 , shuffle=True, callbacks=[early_stop,checkpoint_cb], validation_data=(x_test,y_test),verbose=0)




# In[ ]:


# ```
# predict_val = avg_val + rate * time 
#     east_x = avg_x + rate * time 
#     ...y = avg_y + rate * time 
#     
# 
# avg = predict_val + alpha * ( cluster_val - predict_val)
#     avg_x = est_x + alpha * ( cos(clust_brg) - track_east_x )
#     ..._y = est_y + alpha * ( sin(xxx) - track_east_y) 
#    
# rate_r1 = rate_r0 + beta *( cluster_val - predict_val) 
# ```
np_bearing =  np.loadtxt("./data/bearing.txt")
time_ = 1
cluster_brg = np_bearing[0]

alpha = 0.5 
cluster_brg_rad = np.deg2rad(cluster_brg)

brg_east_x  = cos(cluster_brg_rad)
brg_east_y  = sin(cluster_brg_rad) 
brg_avg_x = cos(cluster_brg_rad)
brg_avg_y = sin(cluster_brg_rad)

brg_rate_x = 1
brg_rate_y = 1

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

predict_east_x = brg_avg_x + brg_rate_x * time_
predict_east_y = brg_avg_y + brg_rate_y * time_


brg_avg_x = predict_east_x + alpha * ()








# In[568]:


cpa_const_v = np.loadtxt("./data/const_v_line_move.txt")
x= np.arange(len(cpa_const_v))
P(cpa_const_v, x)
PS()

c_g = np.gradient(cpa_const_v)

d_rad = np.deg2rad(c_g)


P(d_rad)
PS()

brg_rate = np.log10(np.abs(c_g)+1e-6) + 0.5
P(cpa_const_v, brg_rate)
PS()

P(x, brg_rate)
PS()



cpa = np.load("./data/cpa_raw_.npy")  
idx = 0
P(cpa[idx])
PS()
c_g = np.gradient(cpa[idx])

d_rad = np.deg2rad(c_g)
P(d_rad[:55])
PS()



# In[674]:


def win_smooth(y, window_size=5):
    if not (window_size % 2):
        window_size += 1
    window_half_size = (window_size-1)//2
    weights = np.arange(1,window_size+1).astype(np.float32)
    
    weights /= weights.sum()
    y_smooth = np.convolve(y, weights, mode='valid')
    
    return np.concatenate([y_smooth[0]*np.ones(window_half_size),  y_smooth, y_smooth[-1]*np.ones(window_half_size)])


from scipy import optimize
def fit_func(p,x):
    m,a,b,c=p
    return m*x**3 - a*x**2 + b*x + c

def err_func(p,x,y):
    return fit_func(p,x)-y

def my_leastsq(err_func, x,y):
    p0=[0.01,-0.02,0.03,0]
    params, success = optimize.leastsq(err_func, p0, args=(x,y))
    print("- success: ", success)
    y_smooth = fit_func(params,x)
    return y_smooth

def linear_interp(y, len_y_):
    # from len(y) => len_y_ 
    interp_fn = interp1d(np.arange(len(y)).astype(np.float32), y, kind='linear')
    y_ = interp_fn(np.linspace(0, len(y)-1, len_y_))
    return y_




idx = 42732
start_ = 41
sample_intv = 2


# str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"
# model_id =tf.keras.models.load_model(str_path_save_model)


xn=np.load("./data/xn.npy")
xn=xn[:start_]

y=np.load("./data/y.npy")
y=y[:start_]

y_=np.load("./data/y_.npy")
y_=y_[:start_]

x=np.arange(len(y_)).astype(np.float32)

y_leastsq_ = my_leastsq(err_func, x,y_)

y_s_ = win_smooth(y_, 25)


P(xn)
P(y,c='g')
P(y_)
P(y_leastsq_)
# P(y_s_)

PS()


scale_factor = sample_intv 

y_leastsq_scale_ = linear_interp(y_leastsq_, len(y_leastsq_)//scale_factor)
y_scale_ = linear_interp(y, len(y)//scale_factor)


P(y_leastsq_scale_)
P(y_scale_,c='g')
PS()

c_g_leastsq_scale_ = np.gradient(y_leastsq_scale_)
c_g_scale_ = np.gradient(y_scale_)


P(c_g_leastsq_scale_)
P(c_g_scale_, c='g')

PS()


# y_rate_leastsq_scale_ = np.log10(np.abs(c_g_leastsq_scale_)) + 0.5 
# y_rate_scale_ = np.log10(np.abs(c_g_scale_)) + 0.5 
y_rate_leastsq_scale_ = np.log10(c_g_leastsq_scale_) + 0.5 
y_rate_scale_ = np.log10(c_g_scale_) + 0.5 


P(y_rate_leastsq_scale_)  
P(y_rate_scale_, c='g')
PS()


np.savetxt("./data/y_rate_leastsq_scale_.txt", y_rate_leastsq_scale_)

print("- save txt")

# c_g = np.gradient(y_leastsq_scale_)

# c_g = np.log10(np.abs(c_g)) + 0.5

# P(y_leastsq_scale_ , c_g)

# PS()



# # y_leastsq_back_ = linear_interp(y_leastsq_, start_)

# # P(y_leastsq_back_)
# # PS()






    
    


# In[680]:


cpa_n = np.load("./data/cpa_noise_.npy")
cpa = np.load("./data/cpa_raw_.npy")
x_train, x_test, y_train, y_test  = train_test_split(cpa_n, cpa, test_size=0.2,shuffle=False)



cpa_n[:,start_:] = 0
cpa[:,start_:]=0













# In[1040]:


from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()


def ck_mm(cpa_n, cpa,  idx, start_, mm_dic, max_len):
    id_cpa_n = cpa_n[idx:idx+1,:]
    
    prep_ = min_max_scaler.fit(id_cpa_n.T)
    

    
    id_cpa_n = prep_.transform(id_cpa_n.T).T
    id_cpa_n = id_cpa_n[0]
    
    
    
    id_cpa = cpa[idx:idx+1,:] 
    
    id_cpa = prep_.transform(id_cpa.T).T
    id_cpa = id_cpa[0]
    
    print(id_cpa.shape)

    
    
    
    
    id_cpa_n[start_:] = 0
    id_cpa[start_:] = 0
    
    P(id_cpa_n[:start_])
    P(id_cpa[:start_])
    PS()
    
    
    id_cpa_ = mm_dic[start_].predict(id_cpa_n.reshape(-1,max_len))[0]
    
#     id_cpa_ = id_cpa_ * (max_-min_) + min_
#     id_cpa_n = id_cpa_n * (max_-min_) + min_
#     id_cpa = id_cpa *(max_-min_) + min_
    
    P(id_cpa_n[:start_])
    P(id_cpa[:start_])
    P(id_cpa_[:start_], c='r', label=str(start_))
    plt.legend()
    PS()
    
    

idx = 42799


starti = 19
endi= 20

max_len = 512

mm_dic


# for start_ in np.arange(endi,starti-1,-1):
for start_ in [512]:
    str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"
    mm_dic[start_] =tf.keras.models.load_model(str_path_save_model)



print("- load done")


print (mm_dic.keys())


assert(0)


cpa_n = np.load("./data/cpa_noise_.npy")
cpa = np.load("./data/cpa_raw_.npy")

P(cpa_n[idx,:20])
P(cpa[idx,:20])
PS()


for j in range(1):
    for start_ in np.arange(endi,starti-1,-1):
        print(start_)
        ck_mm(cpa_n, cpa,  idx, start_, mm_dic, max_len)







# In[1161]:


def get_noise(noise,len):
    s_idx = np.random.permutation(len)
    return noise[s_idx]

def to_unit(v):
    # 定义向量v
    v 
    # 计算向量v的模长
    magnitude = np.linalg.norm(v)

    # 计算向量v的单位向量
    unit_vector = v / magnitude

    return unit_vector



np.random.seed(0)
speed_set = np.random.random(10)*0.004 +0.0094

R = 700
uav_loc = np.array([-R/np.sqrt(2), R/np.sqrt(2)])

uav_v = np.array([1,-0.2])
uav_v = to_unit(uav_v)
dt=1
loc_set = np.zeros([512,2])


for step_cnt in np.arange(0,512):
    loc_set[step_cnt,0] = uav_loc[0]
    loc_set[step_cnt,1] = uav_loc[1]
    
    uav_loc += uav_v * dt * speed_set[0];
    
    
# print(loc_set)
    



    
plt.scatter(loc_set[:,0], loc_set[:,1],s=1)
PS()

start_ = 512
x=loc_set[:,0]
y=loc_set[:,1]

# np.abs(atan(y/x))


# np.rad2deg(np.arctan2(-1,1))

noise_sim = np.load("./data/noise_sim.npy")


brg_set = np.rad2deg(np.arctan2(x,y))  


np.save("./data/brg_set_with_minus_raw_.npy", brg_set) 


brg_set_n = brg_set + get_noise(noise_sim, len(brg_set))
np.save("./data/brg_set_with_minus_noise_.npy", brg_set_n) 


min_val = np.min(brg_set_n)
max_val = np.max(brg_set_n)


P(brg_set_n)
P(brg_set)
PS()

brg_set_n = (brg_set_n - min_val) / (max_val - min_val)

brg_set = (brg_set - min_val) / (max_val - min_val)







disp(brg_set.shape)

max_len = 512

display(mm_dic.keys())
brg_set_ = mm_dic[start_].predict(brg_set_n.reshape(-1,max_len))[0,0:start_]

brg_set = brg_set[0:start_]
brg_set_n = brg_set_n[0:start_]


P(brg_set_n)
P(brg_set_ , c='r')
P(brg_set  ,)
PS()


disp(brg_set_.shape)

# P(brg_set_n*(max_val - min_val)+ min_val)
P(brg_set_*(max_val - min_val)+ min_val , c='r')
P(brg_set*(max_val - min_val)+ min_val)
PS()


# ck_mm(cpa_n, cpa,  idx, start_, mm_dic, max_len)





# In[818]:


cpa=np.loadtxt("./data/cpa_raw_.csv", delimiter=",")
scale_factor = 8
cpa_n = np.loadtxt("./data/cpa_noise_.csv", delimiter=",")

start_ = 512
max_len = 512

cpa_ = mm_dic[start_].predict(cpa_n)

y_leastsq_ = my_leastsq(err_func, x,y_)

cpa_r_ =  np.zeros_like(cpa_)

idx=0
for e_ in cpa:
    
    y_=cpa_[idx]
    max_factor = np.max(y_)
    y_ = y_/max_factor
    
    x= np.arange(len(y_)).astype(np.float32)
    y_ = my_leastsq(err_func, x,y_)
    yi_ = linear_interp(y_, len(y_)//scale_factor)
    
    c_g_y_ = np.gradient(yi_)
    y_r_ = np.log10(np.abs(c_g_y_)+1e-6) + 0.5
    yi_r_ = linear_interp(y_r_, len(y_))
    cpa_r_[idx,:] = yi_r_
    P(y_)
    PS()
    P(y_,yi_r_,  c='r')
    
    PS()
    
    idx+=1

cpa_r_


np.savetxt("./data/cpa_p_rate_.csv", cpa_r_, delimiter=",", fmt="%.5f")

print("- save cpa rate ok")  



# In[908]:


import sklearn.preprocessing as prep 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

t = np.linspace(-20,20, 1000).reshape(20,-1)

t_n= np.zeros_like(t)
i=0

for i_ in t:
    t_n[i] = t[i] + get_noise(noise_sim, 50)
    i+=1 

# preprocessor = prep.StandardScaler().fit(t_n)

# t_one = t_n[0]
# prep_ = scaler.fit(t_n.T)

# # t_norm_n =scaler.fit_transform(t_n)
# # t_norm_n =scaler.fit_transform(t_n)

# t_norm_n = prep_.transform(t_n.T).T
# t_norm = prep_.transform(t.T).T

# # t_norm_n = preprocessor.transform(t_n)
# # t_norm = preprocessor.transform(t)


# P(t_n[2])
# P(t[2])
# PS()



# P(t_norm_n[2])
# P(t_norm[2])
# PS()


et_n = pd.DataFrame(t_n[0])
et = pd.DataFrame(t[0])




prep_ = scaler.fit(et_n)

et_norm_n = prep_.transform(et_n)
et_norm = prep_.transform(et)


P(et_n)
P(et)
PS()




et_n_=prep_.inverse_transform(et_norm_n)
et_=prep_.inverse_transform(et_norm)


P(et_n_)
P(et_)
PS()




# P(et_norm_n)
# P(et_norm)
# PS()



id_pd = pd.DataFrame(np.linspace(1,3,6))
id_pd








t
t_n

pd_t_n = pd.DataFrame(t_n.T)
pd_t = pd.DataFrame(t.T)




prep_ = scaler.fit(pd_t_n)
pd_t_norm_n = prep_.transform(pd_t_n).T
pd_t_norm = prep_.transform(pd_t).T



# P(pd_t_norm_n[0])
# P(pd_t_norm[0])
# PS()




prep_ = scaler.fit(t_n.T)
t_norm_n=prep_.transform(t_n.T).T
t_norm=prep_.transform(t.T).T


# P(t_norm_n[0])
# P(t_norm[0])
# PS()





rad_ = np.deg2rad(135)

R*sin(rad_)
R*cos(rad_)









# In[1555]:


cpa = np.load("./data/cpa_uni_raw_train.npy")
cpa_n = np.load("./data/cpa_uni_noise_train.npy") 





idx=32



prep_ = min_max_scaler.fit(cpa_n.T)

cpa_n = prep_.transform(cpa_n.T).T
cpa = prep_.transform(cpa.T).T














x_train, x_test, y_train, y_test  = train_test_split(cpa_n, cpa, test_size=0.2,shuffle=False)

flag_retrain = 0

start_ = 512
epoches = 100
batch_size=256
learning_rate = 1e-4

str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"


model_cpa



# retrain again 
if flag_retrain: # new model from bare
    model_cpa =tf.keras.models.load_model(str_path_save_model)

else:
    model_cpa = tf.keras.Sequential(
    [
    #        _L.BatchNormalization(input_shape=x_train.shape[1:]),

        _L.Dense(units=9, activation="relu", input_shape=x_train.shape[1:]),
        _L.Dropout(0.0223),
        _L.Dense(units=3),


        _L.Dense(units=4),
          _L.Dense(units=8),

        _L.Dense(units=y_train.shape[1])
    ]

    )    


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=5)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
filepath = str_path_save_model,
    save_best_only=True,
    monitor='val_mae',
    mode="min",
    verbose=1
)

model_cpa.compile(loss='mae', optimizer=tf.optimizers.Adam(learning_rate=learning_rate), metrics=['mae'])

model_cpa.summary() 


history_cpa = model_cpa.fit(x_train, y_train, epochs=epoches, batch_size=batch_size , shuffle=True, 
                            callbacks=[early_stop,checkpoint_cb], validation_data=(x_test,y_test),verbose=0)






y_test_ = model_cpa.predict(x_test)




for i in np.random.randint(0,10000,6):
    P(x_test[i])
    P(y_test[i])
    P(y_test_[i], c='r')
    PS()
    



# In[1457]:


def brg_to_brg_rate(brg_set):
    c_g = np.gradient(brg_set)
    brg_rate = np.log10(abs(c_g)+1e-7) + 0.5
    return brg_rate

def to_unit(v):
    magnitude = np.linalg.norm(v)
    unit_vector = v / magnitude
    return unit_vector

def get_angle_degrees(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    angle_in_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees


def get_e_loc_v_speed(loc_set,v_rnd, speed_set,idx):
    return [loc_set[idx,:], v_rnd[idx,:], speed_set[idx]]

def gen_loc_set_for_uav(loc_r0,v_speed, speed_val, max_steps):
    s_ = 0
    dt =1 
    brg_set = np.random.random(max_steps)
    
    for s_ in range(max_steps):
        loc = loc_r0 + s_ * v_speed * speed_val
        x0=loc[0]
        y0=loc[1] 
        # cos => loc x (0,1)
        
        v1 = loc
        v2=np.array([0,1])
        ebrg = get_angle_degrees(v1,v2)
        
        if(x0<0):
            if (y0>0):
                ebrg = 360 - ebrg
                pass
            else:
                ebrg = 360 - ebrg
                pass
            
        
        
        brg_set[s_] = ebrg
    return brg_set

def degree_to_coor(deg):
    if (deg>=270):
        return 3
    if (deg>=180):
        return 2
    if (deg>=90):
        return 1
    return 0

def cal_pass_coor_status(brg_set):
    t=brg_set
    cross_coord_stat = np.array([0,0,0,0])

    tcopy = t.copy()

    if np.any(t<90):
        cross_coord_stat[0] = 1
        
        
    idx = 3
    while idx >= 1:
        
        if np.any(tcopy>90*(idx)):  #4
            cross_coord_stat[idx] = 1
            tcopy[np.where(tcopy>90*idx)] = 0.123456

        idx -= 1
    return cross_coord_stat

def cal_coor_start_end(cross_coord_stat):
    cc = np.concatenate([cross_coord_stat,cross_coord_stat])

    l_c = len(cross_coord_stat)

    coor_num_cross = len( np.ravel(np.where(cross_coord_stat>0)) )

    i = 0
    for i in range(l_c):
        j = i + coor_num_cross
#         print(cc[i:j])
        if np.all(cc[i:j] > 0):
            break

    start = i
    end = i+coor_num_cross-1

    if start == degree_to_coor(t[0]):
#         print("from {} ~ {}".format(start,end))
        pass
    else:
        end,start = start, end
#         print("from {} ~ {}".format(start,end))

    return [start,end]

def uni_degree_brg_set(brg_set):
    cross_coord_stat = cal_pass_coor_status(brg_set)
    
    t=brg_set
    
    [start, end] = cal_coor_start_end(cross_coord_stat)


    if (np.max([start,end]) >=4 ):
        if start < end:
#             print("- cross 360->0 line")
            t=np.where(t<180,t+360,t)
        else:
#             print("- cross 0->360 line")
            t=np.where(t>180,t-360,t)
            

    brg_set_new = t
    return brg_set_new

def gen_init_set_of_loc_v():
    global LEN 
    global R_min 
    global R_max 
    global X_theta_start 
    global X_theta_end 
    global speed_val_start 
    global speed_val_end 
    global max_steps     
    
    loc_set = np.random.random([LEN, 2])
    v_rnd = loc_set.copy()
    speed_set_base = np.random.random(LEN)*0.004 +0.0094
    speed_val_mul_factor = np.random.randint(speed_val_start, speed_val_end, LEN); 

    speed_set = np.multiply(speed_set_base, speed_val_mul_factor)

    idx = 0
    while idx < len(loc_set):
        R = np.random.randint(R_min,R_max)
        theta = np.random.randint(X_theta_start,X_theta_end)+ np.random.random()-0.5 
        rad_ = np.deg2rad(theta)

        loc_set[idx,:] = [R*np.cos(rad_), R*np.sin(rad_)]
        e_v_rnd = [np.random.random()-0.5,np.random.random()-0.5]
        v_rnd[idx,:] = to_unit(e_v_rnd)

        idx+=1
    
    return [loc_set,v_rnd, speed_set]


def gen_sample_brg_set(brg_set, loc_set, v_rnd, speed_set):
    
    step_cnt = 0
    LEN_ = len(brg_set)
    for step_cnt in range(LEN_):
        [loc_r0,v_speed, speed_val] = get_e_loc_v_speed(loc_set,v_rnd, speed_set,step_cnt)

        e_brg_set = gen_loc_set_for_uav(loc_r0,v_speed, speed_val, max_steps)
    #     P(e_brg_set)
    #     PS()

        e_brg_set = uni_degree_brg_set(e_brg_set)
        brg_set[step_cnt] = e_brg_set

    #     P(e_brg_set)
    #     PS()

    print("- done")


    return brg_set


    

### global area ###

global LEN 
global R_min 
global R_max 
global X_theta_start 
global X_theta_end 
global speed_val_start 
global speed_val_end 
global max_steps 
#------------------
LEN = 1000
R_min = 100
R_max = 800
X_theta_start = 0
X_theta_end = 360
speed_val_start = 1
speed_val_end = 155
max_steps = 512
###################

# main_

brg_set = np.zeros([LEN,max_steps], dtype="float32")

[loc_set,v_rnd, speed_set] = gen_init_set_of_loc_v();
brg_set = gen_sample_brg_set(brg_set, loc_set, v_rnd, speed_set)

pd.DataFrame(brg_set).head(4)




# In[1530]:


# re-train , fine-tune , generate 453 models 


### global area ###
global LEN 
global R_MIN 
global R_MAX 
global X_THETA_START 
global X_THETA_END 
global SPEED_VAL_START 
global SPEED_VAL_END 
global MAX_STEPS 
#------------------
LEN = 50
R_MIN = 40
R_MAX = 800

X_THETA_START = 0
X_THETA_END = 360

SPEED_VAL_START = 1
SPEED_VAL_END = 40

MAX_STEPS = 512
###################
global noise_idx 
noise_idx = 0
### global var ###
model_change_data_times = 1
epoches = 1

learning_rate=2e-3
batch_size = 256 
max_len = MAX_STEPS





import tensorflow as tf
import time
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import MinMaxScaler
import copy
import sys

P=plt.plot
PS=plt.show
_L=tf.keras.layers
min_max_scaler = MinMaxScaler()




def brg_to_brg_rate(brg_set):
    c_g = np.gradient(brg_set)
    brg_rate = np.log10(abs(c_g)+1e-7) + 0.5
    return brg_rate

def to_unit(v):
    magnitude = np.linalg.norm(v)
    unit_vector = v / magnitude
    return unit_vector

def get_angle_degrees(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    angle_in_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees


def get_e_loc_v_speed(loc_set,v_rnd, speed_set,idx):
    return [loc_set[idx,:], v_rnd[idx,:], speed_set[idx]]

def gen_loc_set_for_uav(loc_r0,v_speed, speed_val, MAX_STEPS):
    s_ = 0
    dt =1 
    brg_set = np.random.random(MAX_STEPS)
    
    for s_ in range(MAX_STEPS):
        loc = loc_r0 + s_ * v_speed * speed_val
        x0=loc[0]
        y0=loc[1] 
        # cos => loc x (0,1)
        
        v1 = loc
        v2=np.array([0,1])
        ebrg = get_angle_degrees(v1,v2)
        
        if(x0<0):
            ebrg = 360 - ebrg

        
        brg_set[s_] = ebrg
    return brg_set

def degree_to_coor(deg):
    if (deg>=270):
        return 3
    if (deg>=180):
        return 2
    if (deg>=90):
        return 1
    return 0

def cal_pass_coor_status(brg_set):
    t=brg_set
    cross_coord_stat = np.array([0,0,0,0])

    tcopy = t.copy()

    if np.any(t<90):
        cross_coord_stat[0] = 1
        
        
    idx = 3
    while idx >= 1:
        
        if np.any(tcopy>90*(idx)):  #4
            cross_coord_stat[idx] = 1
            tcopy[np.where(tcopy>90*idx)] = 0.123456

        idx -= 1
    return cross_coord_stat

def cal_coor_start_end(cross_coord_stat,first_deg):
    


    cc = np.concatenate([cross_coord_stat,cross_coord_stat])


    l_c = len(cross_coord_stat)

    coor_num_cross = len( np.ravel(np.where(cross_coord_stat>0)) )


    i = 0
    for i in range(l_c):
        j = i + coor_num_cross
        if np.all(cc[i:j] > 0):
            break

    start = i
    end = i+coor_num_cross-1

    if start == degree_to_coor(first_deg):
#         print("from {} ~ {}".format(start,end))
        pass
    else:
        end,start = start, end
#         print("from {} ~ {}".format(start,end))

    return [start,end]

def uni_degree_brg_set(brg_set):
    cross_coord_stat = cal_pass_coor_status(brg_set)
    
    t=brg_set
    first_deg = t[0]

    [start, end] = cal_coor_start_end(cross_coord_stat, first_deg)



    if (np.max([start,end]) >=4 ):
        if start < end:
#             print("- cross 360->0 line")
            t=np.where(t<180,t+360,t)
        else:
#             print("- cross 0->360 line")
            t=np.where(t>180,t-360,t)
            

    brg_set_new = t
    return brg_set_new

def gen_init_set_of_loc_v():
    global LEN 
    global R_MIN 
    global R_MAX 
    global X_THETA_START 
    global X_THETA_END 
    global SPEED_VAL_START 
    global SPEED_VAL_END 
    global MAX_STEPS     
    
    loc_set = np.random.random([LEN, 2])
    v_rnd = loc_set.copy()
    speed_set_base = np.random.random(LEN)*0.004 +0.0094
    speed_val_mul_factor = np.random.randint(SPEED_VAL_START, SPEED_VAL_END, LEN); 

    speed_set = np.multiply(speed_set_base, speed_val_mul_factor)

    idx = 0
    while idx < len(loc_set):
        R = np.random.randint(R_MIN,R_MAX)
        theta = np.random.randint(X_THETA_START,X_THETA_END)+ np.random.random()-0.5 
        rad_ = np.deg2rad(theta)

        loc_set[idx,:] = [R*np.cos(rad_), R*np.sin(rad_)]
        e_v_rnd = [np.random.random()-0.5,np.random.random()-0.5]
        v_rnd[idx,:] = to_unit(e_v_rnd)

        idx+=1
    
    return [loc_set,v_rnd, speed_set]


def gen_sample_brg_set(brg_set):

    [loc_set,v_rnd, speed_set] = gen_init_set_of_loc_v();
    
    
    step_cnt = 0
    LEN_ = brg_set.shape[0] 
    for step_cnt in range(LEN_):

        if step_cnt % 10000 == 0:
            print(step_cnt)

        [loc_r0,v_speed, speed_val] = get_e_loc_v_speed(loc_set,v_rnd, speed_set,step_cnt)
        


        e_brg_set = gen_loc_set_for_uav(loc_r0,v_speed, speed_val, MAX_STEPS)
#         P(e_brg_set)
#         PS()
#         break
#         P(e_brg_set)
#         PS()
        
        e_brg_set = uni_degree_brg_set(e_brg_set)
#         P(e_brg_set)
#         PS()
    

        brg_set[step_cnt,:] = e_brg_set
#         P(e_brg_set)
#         PS()

        

    #     P(e_brg_set)
    #     PS()

    print("- done")


    return brg_set




# main_


cpa = np.load("./data/cpa_raw_.npy")
# noise_sim_r0 = np.load("./data/noise_sim.npy")

brg_set = np.zeros([LEN,MAX_STEPS], dtype="float32")

cpa=brg_set

cpa.shape

gen_sample_brg_set(cpa)



# print(cpa[0])

for i in range(9):
    P(cpa[i])

PS()

# for i in range(9):
#     P(brg_to_brg_rate(cpa[i]))

# PS()


# In[1572]:


def get_noise(noise,len):
    s_idx = np.random.permutation(len)
    return noise[s_idx]

noise_sim = np.load("./data/noise_sim.npy")


cpa = np.load("./data/cpa_uni_raw_train.npy")
cpa_n = cpa.copy()

step_cnt = 0
for e in cpa:
    cpa_n[step_cnt] = cpa[step_cnt] + get_noise(noise_sim, 512)
    step_cnt+=1
for i in range(9):
    P(cpa_n[i])
    P(cpa[i])
    PS()
    
np.save("./data/cpa_uni_noise_train.npy", cpa_n)   
# np.save("./data/cpa_uni_raw_inf.npy", cpa)      

print("- save ok")

    


# In[1549]:


cpa = np.load("./data/cpa_uni_raw_train.npy")
cpa_n = np.load("./data/cpa_uni_noise_train.npy") 
start_ = 512

str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"
mm_dic[start_] =tf.keras.models.load_model(str_path_save_model)


for i in np.random.randint(0,1000,9):
    P(cpa_n[-i])
    P(cpa[-i])
PS()


prep_ = min_max_scaler.fit(cpa_n.T)

cpa_n = prep_.transform(cpa_n.T).T
cpa = prep_.transform(cpa.T).T



for i in np.random.randint(0,1000,1):
    P(cpa_n[-i])
    P(cpa[-i])
PS()





cpa_ = mm_dic[start_].predict(cpa_n)


cpa_n = prep_.inverse_transform(cpa_n.T).T
cpa = prep_.inverse_transform(cpa.T).T
cpa_ = prep_.inverse_transform(cpa_.T).T


for i in np.random.randint(0,1000,1):
    P(cpa_n[-i])
    P(cpa_[-i], c='r')
    P(cpa[-i])
    PS()
# PS()














# In[1570]:


def linear_interp(y, len_y_):
    # from len(y) => len_y_ 
    interp_fn = interp1d(np.arange(len(y)).astype(np.float32), y, kind='linear')
    y_ = interp_fn(np.linspace(0, len(y)-1, len_y_))
    return y_


def ck_mm(model_, edata_n, start_):
    max_len = len(edata_n)
    assert(max_len>=start_)
    
    min_ = np.min(edata_n[:start_])
    max_ = np.max(edata_n[:start_])
    edata_n_old = edata_n.copy()
    
    edata_n[:start_] = (edata_n[:start_] - min_)/ (max_ - min_)
    edata_n[start_:] = 0
    
    edata_ = model_.predict(edata_n.reshape(1,max_len))[0]
    edata_ = edata_ * (max_-min_) + min_
    return [edata_n_old[:start_], edata_[:start_]] 


# start_ = 21
# str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"

# max_len = 512

# print(str_path_save_model)

# # retrain again 

# model_cpa =tf.keras.models.load_model(str_path_save_model)
# mm_dic[start_] = model_cpa

# display(mm_dic.keys())
# cpa_n = np.load("./data/cpa_noise_.npy")
# cpa = np.load("./data/cpa_raw_.npy")





# cpa_n = cpa_n[:,:start_]
# cpa = cpa[:,:start_]

# prep_ = min_max_scaler.fit(cpa_n.T)
# cpa_n = prep_.transform(cpa_n.T).T
# cpa   = prep_.transform(cpa.T).T



# P(cpa_n[-1])
# P(cpa[-1])
# PS()


# cpa_n = tf.keras.preprocessing.sequence.pad_sequences(cpa_n, value=0.0,padding='post',dtype="float32", maxlen=max_len)
# cpa   = tf.keras.preprocessing.sequence.pad_sequences(cpa  , value=0.0,padding='post', dtype="float32", maxlen=max_len)

# disp(cpa_n.shape)




# cpa_ = model_cpa.predict(cpa_n)

# for idx in np.random.randint(1,10000,10):
#     break
#     idx = idx * (-1)

#     P(cpa_n[idx,:start_])
#     P(cpa_[idx,:start_], c='r')
#     P(cpa[idx,:start_])
#     PS()



# cpa_n = prep_.inverse_transform(cpa_n.T).T
# cpa_ = prep_.inverse_transform(cpa_.T).T
# cpa = prep_.inverse_transform(cpa.T).T




# # P(cpa_n[idx])
# P(cpa_[idx], c='r')
# P(cpa[idx])
# PS()


start_=50


str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"
model_cpa =tf.keras.models.load_model(str_path_save_model)
mm_dic[start_] = model_cpa

cpa = np.load("./data/cpa_uni_raw_inf.npy")
cpa_n = np.load("./data/cpa_uni_noise_inf.npy") 


    






for idx in np.random.randint(1,10000,11):
    len_ = len(cpa[0])
    edata_n = cpa_n[-idx]
    edata = cpa[-idx]
    
    edata_n = linear_interp(edata_n[:start_], len_)
    edata = linear_interp(edata[:start_], len_)
    
    
    
    

    [edata_n, edata_] = ck_mm(mm_dic[512], edata_n, len(cpa[0])) 


    P(edata_n)
    P(edata_, c='r')
    P(edata)
    PS()




    


# In[1694]:


def load_input_dim_model(input_dim, mm_dic, model_input_list):

#     input_dim = 20
#     model_input_list = [20,30,40,50,60,80,100,160,300,400,512]

    for start_ in model_input_list:
        if start_>=input_dim:
            break
    assert(start_>0 and start_ <=512)
    
    str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"
    
    # always reload 
    
    
    if start_ in mm_dic.keys():
        print("- use cached model cpa")
        pass
    else:
        model_cpa =tf.keras.models.load_model(str_path_save_model)
        mm_dic[start_] = model_cpa
    
    
    return start_

def resize_list(y, len_y_):
    # from len(y) => len_y_ 
    interp_fn = interp1d(np.arange(len(y)).astype(np.float32), y, kind='linear')
    y_ = interp_fn(np.linspace(0, len(y)-1, len_y_))
    return y_


def ck_mm(model_, edata_n):
    assert(model_.output_shape[-1] == len(edata_n))
    
    edata_ = model_.predict(edata_n.reshape(-1,len(edata_n)))[0]
    
    min_ = np.min(edata_n)
    max_ = np.max(edata_n)
    edata_n_old = edata_n.copy()
    
    edata_n = (edata_n - min_)/ (max_ - min_)
    
    
    edata_ = model_.predict(edata_n.reshape(1,len(edata_n)))[0]
    
    edata_s_ = leastsq_smooth(edata_)

    
    edata_ = edata_ * (max_-min_) + min_
    edata_s_ = edata_s_ * (max_-min_)+min_
    
    return [edata_n_old, edata_, edata_s_] 
    
    
def predict_from_e_shape(edata_n, mm_dic, model_inputshape_list):
    input_dim = len(edata_n)
    edata_n_old = edata_n.copy()

    start_ =  load_input_dim_model(input_dim, mm_dic, model_inputshape_list)

    
    
    edata_n = resize_list(edata_n, start_)

    [edata_n, edata_, edata_s_] = ck_mm(mm_dic[start_], edata_n)   
    
#     edata_n = resize_list(edata_n,input_dim)
    edata_n = edata_n_old
    edata_ = resize_list(edata_,input_dim)
    edata_s_ = resize_list(edata_s_,input_dim)
    
    return [edata_, edata_s_]



def fit_func(p,x):
    m,a,b,c=p
    return m*x**3 - a*x**2 + b*x + c

def err_func(p,x,y):
    return fit_func(p,x)-y

def my_leastsq(err_func, x,y):
    p0=[0.01,-0.02,0.03,0]
    params, success = optimize.leastsq(err_func, p0, args=(x,y))
    print("- success: ", success)
    y_smooth = fit_func(params,x)
    return y_smooth

def leastsq_smooth(y):
    x = np.arange(len(y)).astype(np.float32)
    y_smooth = my_leastsq(err_func,x,y)
    return y_smooth



    

cpa = np.load("./data/cpa_uni_raw_inf.npy")
cpa_n = np.load("./data/cpa_uni_noise_inf.npy") 

idx=np.random.randint(0,50000, 1)[0]
edata_n=cpa_n[idx,:21]
edata=cpa[idx,:21]

# mm_dic={}

input_dim = len(edata_n)
model_inputshape_list = [20,30,40,50,60,80,100,160,300,400,512]

print(mm_dic.keys())

P(edata_n)
# P(edata_,c='r')
P(edata)
PS()


[edata_, edata_s_] = predict_from_e_shape(edata_n, mm_dic, model_inputshape_list)



P(edata_n)
P(edata_,c='r')
P(edata_s_,c='black')
P(edata)
PS()








# In[1687]:


# e=cpa_n[1]

# e=e[:55]
# e_resize = resize_list(e,77)
# P(e)
# P(e_resize)
# PS()

# e_resize_resize = resize_list(e_resize,55)

# P(e)
# P(e_resize_resize,c='r')
# PS()



# i_dic = {'a':0, 'b':1}

# if "a" in i_dic.keys():
#     del i_dic['a']


# i_dic













# In[21]:


def gen_data_clip_idx_pair(start_, max_len):
    start_ = 140
    for start__ in range(start_,max_len-start_, start_):
        print(start__, start__+start_)

    print(max_len-start__, max_len)

gen_data_clip_idx_pair(140,512)




import tensorflow as tf
import time
from math import atan2, pi, cos,sin
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as disp
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import MinMaxScaler
import copy
import sys
import pandas as pd 


P=plt.plot
PS=plt.show
_L=tf.keras.layers
min_max_scaler = MinMaxScaler()




"./data/"+str(time.strftime("%Y-%m-%d_%H%M%S", time.localtime()))+".npy"





t=np.array([140,512,400,300,160,100,80,60,50,40,30,25,20,480,450,350,290,230,180])
#t=np.array([180,230,290,350,450,480,25,160,300,    20,25,30,40,50,60,80,100,160,300,400,    512,140])

np.sort(t)








# In[73]:


from scipy.interpolate import interp1d


def resize_list(y, len_y_):
    # from len(y) => len_y_ 
    interp_fn = interp1d(np.arange(len(y)).astype(np.float32), y, kind='linear')
    y_ = interp_fn(np.linspace(0, len(y)-1, len_y_))
    return y_

def get_noise(noise,len):
    s_idx = np.random.permutation(len)
    return noise[s_idx]


def brg_to_brg_rate(brg_set, sample_factor):
    len_ = len(brg_set)
    brg_set_ = resize_list(brg_set, len(brg_set)//sample_factor)

    c_g = np.gradient(brg_set_)
    brg_rate = np.log10(np.abs(c_g)+1e-7) + 0.5
    brg_rate = resize_list(brg_rate, len_)

    return brg_rate

def brg_to_brg_rate_no_log10(brg_set,sample_factor):
    len_ = len(brg_set)
    
    if sample_factor == 1:
        return np.abs(np.gradient(brg_set))
    
    brg_set_ = resize_list(brg_set, len(brg_set)//sample_factor)

    c_g = np.gradient(brg_set_)
    brg_rate__ = np.abs(c_g)
    brg_rate___ = resize_list(brg_rate__, len_)
    return brg_rate___


cpa=np.load("./data/cpa_uni_raw_train.npy")
cpa_n=np.load("./data/cpa_uni_noise_train.npy")






cpa_rate = cpa.copy()


idx = 0
for e_ in cpa:
    cpa_rate[idx] = brg_to_brg_rate_no_log10(e_,1)
    idx+=1

# np.random.seed(19)


"""
for i in np.random.randint(40000,50000,4):
    P(cpa_rate[i], c='r')
    PS()
    P(cpa[i])
    PS()

"""


P(cpa_n[0])

PS()

cpa_avg=cpa_n[0] - np.median(cpa_n[0])


P(cpa_avg)

PS()




t=np.array([1,2,9],[4,5,7])
np.median(t)




# In[242]:


from scipy import optimize
from sklearn.linear_model import Lasso




def lasso_model(y,alpha=0.2):
    pad_num = 11
    y_len_old = len(y)
    
    y=np.pad(y, (pad_num, pad_num), 'symmetric', reflect_type='odd' )
    x = np.arange(len(y))
    
    x=x.reshape(-1,1)
#     X=np.column_stack((np.power(x,0), x, np.power(x,2), np.power(x,3), np.power(x,4)))
    X=np.column_stack(( x, np.power(x,2), np.power(x,3)))
    lasso = Lasso(alpha=alpha)
    lasso.fit(X,y)
    y_pred = lasso.predict(X)
    lasso.coef_
    return y_pred[pad_num:y_len_old+pad_num]




def reshape_2_timesteps_feature(data_1d, time_steps, features):
    data_2d=np.array([data_1d])
    assert(time_steps*features == data_2d.shape[1])
    data_3d = data_2d.reshape(data_2d.shape[0], time_steps, features)
    return data_3d


def fit_func(p,x):
    m,a,b,c=p
    return m*x**3 - a*x**2 + b*x + c

def err_func(p,x,y):
    return fit_func(p,x)-y

def my_leastsq(err_func, x,y):
    p0=[0.01,-0.02,0.03,0]
    params, success = optimize.leastsq(err_func, p0, args=(x,y))
    print("- success: ", success)
    y_smooth = fit_func(params,x)
    return y_smooth

def leastsq_smooth(y):
    x = np.arange(len(y)).astype(np.float32)
    y_smooth = my_leastsq(err_func,x,y)
    return y_smooth




def brg_to_brg_rate_no_log10(brg_set,sample_factor):
    len_ = len(brg_set)
    
    if sample_factor == 1:
        return np.abs(np.gradient(brg_set))
    
    brg_set_ = resize_list(brg_set, len(brg_set)//sample_factor)

    c_g = np.gradient(brg_set_)
    brg_rate__ = np.abs(c_g)
    brg_rate___ = resize_list(brg_rate__, len_)
    return brg_rate___


def ck_mm_rate(model_, edata_n):
    assert(model_.output_shape[-1] == len(edata_n))
    time_steps=64
    features = int(len(edata_n)/time_steps)

    
#     edata_ = model_.predict(edata_n.reshape(-1,len(edata_n)))[0]

    
    mean_ = np.mean(edata_n)
#     max_ = np.max(edata_n)
    edata_n_old = edata_n.copy()
    
    edata_n = edata_n - mean_
    
    
    edata_rate_ = model_.predict( reshape_2_timesteps_feature(edata_n, time_steps, features))[0]
    
#     edata_rate_s_ = leastsq_smooth(edata_rate_)
    edata_rate_s_ = lasso_model(edata_rate_,alpha=0.2)

    
    return [edata_n_old, edata_rate_, edata_rate_s_] 



cpa=np.load("./data/cpa_uni_raw_inf.npy")
cpa_n= np.load("./data/cpa_uni_noise_inf.npy")
time_steps = 64
features = int(cpa.shape[1]/time_steps)


start_= 512
str_path_save_model = "./model_cpa/cpa_rate-i-"+format(start_,"04")+".h5"
model = tf.keras.models.load_model(str_path_save_model)

model.summary()


for idx in np.random.randint(10000,40000,20):
# for idx in [13515,13479]:
    print(idx)
    edata_n=cpa_n[idx]
    edata=cpa[idx]
    
    [edata_n, edata_rate_, edata_rate_s_] = ck_mm_rate(model, edata_n)
    edata_rate = brg_to_brg_rate_no_log10(edata, 1)
    
    P(edata_rate)
#     P(edata_rate_s_,c='r')
    P(edata_rate_,c='r')
    PS()
    
    
    










# In[210]:


# def_ start

def fit_func(p,x):
    m,a,b,c=p
    return m*x**3 - a*x**2 + b*x + c

def err_func(p,x,y):
    return fit_func(p,x)-y

def my_leastsq(err_func, x,y):
    p0=[0.01,-0.02,0.03,0]
    params, success = optimize.leastsq(err_func, p0, args=(x,y))
#     print("- success: ", success)
    y_smooth = fit_func(params,x)
    return y_smooth

def leastsq_smooth(y):
    x = np.arange(len(y)).astype(np.float32)
    y_smooth = my_leastsq(err_func,x,y)
    return y_smooth


def brg_to_brg_rate(brg_set, sample_factor):
    len_ = len(brg_set)
    brg_set_ = resize_list(brg_set, len(brg_set)//sample_factor)

    c_g = np.gradient(brg_set_)
    brg_rate = np.log10(np.abs(c_g)+1e-7) + 0.5
    brg_rate = resize_list(brg_rate, len_)

    return brg_rate

# need_ start
def uni_degree_brg_set(brg_set):  # need_
    cross_coord_stat = cal_pass_coor_status(brg_set)
    
    t=brg_set
    first_deg = t[0]

    [start, end] = cal_coor_start_end(cross_coord_stat, first_deg)



    if (np.max([start,end]) >=4 ):
        if start < end:
#             print("- cross 360->0 line")
            t=np.where(t<180,t+360,t)
        else:
#             print("- cross 0->360 line")
            t=np.where(t>180,t-360,t)
            

    brg_set_new = t
    return brg_set_new

def degree_to_coor(deg):  # need_
    if (deg>=270):
        return 3
    if (deg>=180):
        return 2
    if (deg>=90):
        return 1
    return 0

def cal_pass_coor_status(brg_set):  # need_
    t=brg_set
    cross_coord_stat = np.array([0,0,0,0])

    tcopy = t.copy()

    if np.any(t<90):
        cross_coord_stat[0] = 1
        
        
    idx = 3
    while idx >= 1:
        
        if np.any(tcopy>90*(idx)):  #4
            cross_coord_stat[idx] = 1
            tcopy[np.where(tcopy>90*idx)] = 0.123456

        idx -= 1
    return cross_coord_stat

def cal_coor_start_end(cross_coord_stat,first_deg): # need_
    


    cc = np.concatenate([cross_coord_stat,cross_coord_stat])


    l_c = len(cross_coord_stat)

    coor_num_cross = len( np.ravel(np.where(cross_coord_stat>0)) )


    i = 0
    for i in range(l_c):
        j = i + coor_num_cross
        if np.all(cc[i:j] > 0):
            break

    start = i
    end = i+coor_num_cross-1

    if start == degree_to_coor(first_deg):
#         print("from {} ~ {}".format(start,end))
        pass
    else:
        end,start = start, end
#         print("from {} ~ {}".format(start,end))

    return [start,end]



# need_ end 




def load_input_dim_model(input_dim, mm_dic, model_input_list):

#     input_dim = 20
#     model_input_list = [20,30,40,50,60,80,100,160,300,400,512]

    for start_ in model_input_list:
        if start_>=input_dim:
            break
    assert(start_>0 and start_ <=512)
    
    str_path_save_model = "./model_cpa/cpa-i-"+format(start_,"04")+".h5"
    
    # always reload 
    
    
    if start_ in mm_dic.keys():
        #print("- use cached model cpa")
        pass
    else:
        model_cpa =tf.keras.models.load_model(str_path_save_model)
        mm_dic[start_] = model_cpa
    
    
    return start_

def resize_list(y, len_y_):
    # from len(y) => len_y_ 
    interp_fn = interp1d(np.arange(len(y)).astype(np.float32), y, kind='linear')
    y_ = interp_fn(np.linspace(0, len(y)-1, len_y_))
    return y_


def ck_mm(model_, edata_n):
    assert(model_.output_shape[-1] == len(edata_n))
    
    #edata_ = model_.predict(edata_n.reshape(-1,len(edata_n)))[0]
    
    min_ = np.min(edata_n)
    max_ = np.max(edata_n)
    edata_n_old = edata_n.copy()
    
    edata_n = (edata_n - min_)/ (max_ - min_)
    
    
    edata_ = model_.predict(edata_n.reshape(1,len(edata_n)), verbose=0)[0]
    
    edata_s_ = leastsq_smooth(edata_)

    
    edata_ = edata_ * (max_-min_) + min_
    edata_s_ = edata_s_ * (max_-min_)+min_
    
    return [edata_n_old, edata_, edata_s_] 
    
    
def predict_from_e_shape(edata_n, mm_dic, model_inputshape_list):
#     global flag_print
    input_dim = len(edata_n)
    edata_n_old = edata_n.copy()

    start_ =  load_input_dim_model(input_dim, mm_dic, model_inputshape_list)

#     if flag_print == 0:
#         print("- pick model idx: ", start_)
#     flag_print += 1

    
    
    edata_n = resize_list(edata_n, start_)

    [edata_n, edata_, edata_s_] = ck_mm(mm_dic[start_], edata_n)   
    
#     edata_n = resize_list(edata_n,input_dim)
    edata_n = edata_n_old
    edata_ = resize_list(edata_,input_dim)
    edata_s_ = resize_list(edata_s_,input_dim)
    
    return [edata_, edata_s_]

# def_ end







    
    



def e_lim(e_cpa_first, min_old,max_old):
    max_ = np.max(e_cpa_first)
    min_ = np.min(e_cpa_first)
    intv = 2.5
    if max_ >= max_old:
        max_old = max_ + intv
    if min_ < min_old:
        min_old = min_ - intv
    return [min_old, max_old]
 
    
t=np.eye(4)
from IPython import display

cpa=np.load("./data/cpa_uni_raw_inf.npy")
cpa_n= np.load("./data/cpa_uni_noise_inf.npy")

idx = 15
edata = cpa[idx]
edata_n = cpa_n[idx]

model_inputshape_list = [ 20, 25,  30,  40,  50,  60,  80, 100, 140, 160, 180, 230, 290, 300, 350, 400, 450, 480, 512]     # total 19 

mm_dic = {}
    
    
"""
max_ = np.max(edata_n[:10])+1.5
min_ = np.min(edata_n[:10])-1.5

for i in range(10,len(e_cpa)):
    display.clear_output(wait=True)
    [min_,max_] = e_lim(edata_n[:i], min_, max_)
    plt.xlim((min_,max_))
    plt.ylim((-5,i+10))
    plt.xlabel('bearings')
    plt.ylabel("time")

    
    ed_n = edata_n[:i]
    x = np.arange(len(ed_n))
    [edata_, edata_s_] = predict_from_e_shape(ed_n, mm_dic, model_inputshape_list)
    
    plt.scatter(edata_n[:i],x, alpha=0.3, label='observed')
    P(edata[:i],x, c='y')
    P(edata_,x, c='r', label="ml")
    P(edata_s_,x, c='black', label="ml+smooth")
    plt.legend()
    
    plt.pause(1e-11)

    

PS()

print("- show end")

"""   
