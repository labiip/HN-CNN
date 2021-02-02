import numpy as np
import random
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os
import warnings
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle  
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from numpy import interp
import matplotlib.pyplot as plt
import matplotlib as mpl
from CV_XGB import CV_XGB
mpl.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

#Extract known or unknown index
def Original_data(ASD):
    ASD_arr = ASD.values.flatten()
    ASD_mask=np.ones_like(ASD)
    ASD_mask[np.isnan(ASD.values)]=0
    ASD_mask_arr = ASD_mask.flatten()
    known_mask = np.nonzero(ASD_mask_arr)[0]   #已知数的index
    unknown_mask = np.delete(range(len(ASD_arr)),known_mask)#未知数的index
    return ASD_mask_arr,known_mask,unknown_mask

#generate negative data
def Generate_negative_values(ASD):
    (m,n) = ASD.shape
    ASD_arr = ASD.values.flatten()
    ASD_mask=np.ones_like(ASD)
    ASD_mask[np.isnan(ASD.values)]=0
    ASD_mask_arr = ASD_mask.flatten()
    pos_mask = np.nonzero(ASD_mask_arr)[0]   #已知数的index
    unknown_mask = np.delete(range(len(ASD_arr)),pos_mask)#未知数的index
    neg_index = np.array(random.sample(list(unknown_mask),len(pos_mask)))
    # ASD_mask_arr[neg_index] = 1
    # ASD_mask = ASD_mask_arr.reshape(m,n)
    return ASD_mask,pos_mask,neg_index

def train_id_to_mat(pos_index,neg_index,num_disease,Site_row,Disease_row):
    pos_site = np.int32(pos_index/num_disease)
    pos_disease = np.int32(pos_index%num_disease)
    neg_site = np.int32(neg_index/num_disease)
    neg_disease = np.int32(neg_index%num_disease)
    pos_mat = np.hstack((Site_row[pos_site],Disease_row[pos_disease]))
    neg_mat = np.hstack((Site_row[neg_site],Disease_row[neg_disease]))
    all_mat = np.vstack((pos_mat,neg_mat))
    labels = np.hstack((np.ones(len(pos_index)),np.zeros(len(neg_index))))
    all_mat,labels = shuffle(all_mat,labels)
    all_mat = all_mat.reshape(-1,2,918,1)
    labels = labels.reshape(-1,1)
    return all_mat,labels

#定义placeholder占位符
xs = tf.placeholder(tf.float32,[None,2,918,1],name = "x_place")
ys = tf.placeholder(tf.float32,[None,1],name = "y_place")

#定义卷积核
def init_filter(shape):
    filters =  tf.Variable(tf.random_normal(shape,dtype=tf.float32))
    return (filters)
    
#定义卷积层
def conv2d(x,filters):
    # stride[1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] =1
    #strides[0]和strides[3]的两个1是默认值，意思是不对样本个数和channel进行卷积
    #中间两个1代表padding是在x方向运动一步，y方向运动一步
    return tf.nn.conv2d(x, filters, strides=[1,1,1,1], padding="SAME")

def max_pool(x):
    #池化的核函数大小为2*2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def random_mini_batches(X,Y,mini_batch_size,seed=0):
    np.random.seed(seed) #指定随机种子
    m = X.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m)) 
    # Y = Y.reshape(-1,1)
    shuffled_X = X[permutation,:] 
    shuffled_Y = Y[permutation,:].reshape((-1,1))
    num_complete_minibatches = np.math.floor(m / mini_batch_size) 
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_size * num_complete_minibatches:,:]
        mini_batch_Y = shuffled_Y[mini_batch_size * num_complete_minibatches:,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

#定义模型
with tf.name_scope("Model"):
    filter1 = init_filter([2,4,1,32])#[16,16,1,32]
    b_conv1 = init_filter([1])

#def model(x):
    
    #Convolution &Dropout(0.5) & Max Pooling ————Layer1
    h_conv1 = tf.nn.relu(conv2d(xs,filter1)+b_conv1)
    #h_dp1 = tf.nn.dropout(h_conv1,0.5)
    h_pool1 = max_pool(h_conv1)
    #Flatten1 & Dense1 ————Layer2
    h_flatten1 = tf.layers.flatten(h_pool1)
    print('h_flatten1',h_flatten1)
#    dense1 = add_layer(h_flatten1,w0,b0,activation_function=tf.nn.relu)
    dense1 = tf.nn.relu(tf.layers.dense(h_flatten1, 256))
    #Dropout(0.5) & Dense 
    #h_dp2 = tf.nn.dropout(dense1,0.8)
    logits = tf.layers.dense(inputs=dense1, units=1)
    y_vector = tf.nn.sigmoid(logits)
    # y_vector = tf.arg_max(logits,1)
    print('y_vector',y_vector)

with tf.name_scope("LossFunction"):
    # cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=dense1,labels=ys)
    # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dense1,labels=ys))
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense1,labels=ys)
    cross_entropy = tf.reduce_mean(tf.keras.losses.binary_crossentropy(ys,y_vector))
    # cross_entropy=tf.reduce_sum(-(ys*tf.log(y_vector)+(1-ys)*tf.log(1-y_vector)))  
    
#optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#ADAM


#Input
ASD = pd.read_excel('m7G_Data/Site_Disease_mat.xlsx', header=None)#741site*177disease
(num_site,num_disease) = ASD.shape
ASD_arr,known_mask,unknown_mask = Original_data(ASD)
# ASD,pos_index,neg_index = Generate_negative_values(ASD)
ASS_sim = pd.read_excel('m7G_Data/Site_Jaccard_Sim_mat.xlsx', header=None)#741site
ADD_sim = pd.read_excel('m7G_Data/Disease_Sim_mat_V1.xlsx', header=None)  #177disease

# A_Site_row = np.hstack((ASS_sim,ASD)) 
# A_Disease_row = np.hstack((ASD.T,ADD_sim))
# A = np.vstack((A_Site_row,A_Disease_row))

# # pos_site = np.int32(pos_index/num_disease)
# # pos_disease = np.int32(pos_index%num_disease)
# pos_site,pos_disease,neg_site,neg_disease = index_to_code(pos_index,neg_index,num_disease)
# pos_mat = np.hstack((A_Site_row[pos_site],A_Disease_row[pos_disease]))
# neg_mat = np.hstack((A_Site_row[neg_site],A_Disease_row[neg_disease]))

# #正负样本合并起来之后在进行fold
# mat = np.vstack((pos_mat,neg_mat))
# labels = np.append(np.ones(len(pos_index)),np.zeros(len(neg_index)))
# labels = labels.reshape(-1,1)

kf = KFold(n_splits =10, shuffle=True)
train_all=[]
test_all=[]
for train_ind,test_ind in kf.split(range(len(known_mask))):
    train_all.append(train_ind) 
    test_all.append(test_ind) 


tprs = []
aucs = []
loss_data = []
mean_fpr = np.linspace(0, 1, 100)
epochs=50
batch_size=30

for fold_int in range(10):
    print('fold_int',fold_int)
    train_id = train_all[fold_int]
    test_id = test_all[fold_int]
    
    Res_arr = np.zeros_like(ASD_arr)#生成新的关联矩阵
    pos_index = known_mask[train_id]
    Res_arr[pos_index] = 1
    neg_index = np.array(random.sample(list(unknown_mask),len(train_id)))#采用和正样本数量一样的负样本
    # Res_arr[neg_index] = 1
    
    
    Res_mat = Res_arr.reshape(num_site,num_disease)#用于拼接的训练矩阵
    A_Site_row = np.hstack((ASS_sim,Res_mat)) 
    A_Disease_row = np.hstack((Res_mat.T,ADD_sim))
    
    x_train,y_train = train_id_to_mat(pos_index,neg_index,num_disease,
                                    A_Site_row,A_Disease_row)
    
    pos_index = known_mask[test_id]
    neg_index_t = np.delete(unknown_mask,neg_index)
    neg_index = np.array(random.sample(list(neg_index_t),77))#len(test_id)
    x_test,y_test = train_id_to_mat(pos_index,neg_index,num_disease,
                                    A_Site_row,A_Disease_row)
    # pos_site = np.int32(pos_index/num_disease)
    # pos_disease = np.int32(pos_index%num_disease)
    # x_test = np.hstack((A_Site_row[pos_site],A_Disease_row[pos_disease]))
    # x_test = x_test.reshape(-1,2,918,1)
    # y_test = np.ones(len(test_id))
    # y_test = y_test.reshape(-1,1)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        train_batches = random_mini_batches(x_train, y_train,mini_batch_size=batch_size)
        
        #构建循环迭代
        step = 0
        for e in range(epochs):
            for batch_x, batch_y in train_batches:
            # e、执行模型优化
                sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y})
                
                # t_loss = sess.run(cross_entropy,feed_dict={xs: x_train, ys: y_train})
                # loss_data.append(t_loss)
            
                if step % 1000 == 0:
                    print('loss',sess.run(cross_entropy,feed_dict={xs: x_train, ys: y_train}))
                step+=1
        print('last_loss',sess.run(cross_entropy,feed_dict={xs: x_train, ys: y_train}))
        
        x_train_XGB = sess.run(dense1,feed_dict={xs:x_train})
        y_train_XGB = y_train
        x_test_XGB = sess.run(dense1,feed_dict={xs:x_test})
        y_test_XGB = y_test
        
    dtrain = xgb.DMatrix(x_train_XGB,label = y_train_XGB)
    dtest = xgb.DMatrix(x_test_XGB)
    
    params = CV_XGB(x_train_XGB,y_train_XGB)
    watchlist = [(dtrain,'train')]
    
    bst=xgb.train(params,dtrain,num_boost_round=300,evals=watchlist,early_stopping_rounds=100)
    
    Y_predict=bst.predict(dtest)

        
    fpr,tpr,threshold=roc_curve(y_test_XGB,Y_predict,pos_label=1)
    interp_y=interp(mean_fpr, fpr, tpr)
    tprs.append(interp_y)      
    tprs[-1][0]=0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print('roc',roc_auc)
    #aucs.append(roc_auc)
    plt.plot(fpr,tpr,lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (fold_int, roc_auc))
        
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)#label='Chance',
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)   #auc of mean
    std_auc = np.std(aucs)        #5 auc      
    print('mean_auc',mean_auc)
    print('std_auc',std_auc) 
    #integration all ROCs 
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC_cruve")  #title
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('ROC_cruve.png')   #savefig_title
    
aucs.append(mean_auc)
aucs.append(std_auc)
print('aucs',aucs)










