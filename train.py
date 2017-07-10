#coding:utf-8
import tensorflow as tf
from ops import *
import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import  Image
import pandas as pd
#Parameters set
sample_size = 98000
batch_size = 100
epoch = 5
imglist = np.loadtxt('input.txt',dtype='string',delimiter=' ')
image_path = 'data/'
pose_dict = dict(zip(['041','050','051','080','130','140','190'],range(7)))
#Placeholder set
inputs = tf.placeholder(tf.float32,[batch_size,60,60,1],name='real_images')
y = tf.placeholder(tf.int64, [batch_size], name='y')
z = tf.placeholder(tf.float32, [batch_size,50], name='z')
dis_pose = tf.placeholder(tf.int64, [batch_size], name='pose')
gen_pose = tf.placeholder(tf.int64, [batch_size], name='pose')
gen_pose_onehot = tf.one_hot(gen_pose,7)
#Fetch images
def get_batch(idx):
    temp = imglist[idx:idx+batch_size]
    batch_images = np.zeros((batch_size,60,60,1))
    for i in range(temp.shape[0]):
        img = Image.open(image_path+temp[i,0])
        array_img = np.array(img)
        batch_images[i,:,:,0] = (array_img - array_img.mean())/array_img.std()
    batch_labels = temp[:,1].astype('int')
    temp=pd.DataFrame(temp)
    batch_dis_pose = temp[3].apply(lambda x:pose_dict[x]).values
    batch_gen_pose = temp[2].apply(lambda x:pose_dict[x]).values
    return batch_images,batch_labels-1,batch_dis_pose,batch_gen_pose
#Construction of generator and discriminator
def generator(z, image,pose,is_train,reuse=False):
    with tf.variable_scope("Generator",reuse=reuse) as scope:
        h10 = tf.nn.elu(bn(conv2d(image,filters=32,kernel_size=3),is_train))
        h11 = tf.nn.elu(bn(conv2d(h10,filters=64,kernel_size=3),is_train))
        h12 = tf.nn.elu(bn(conv2d(h11,filters=64,kernel_size=3,strides=[2,2]),is_train))      
        h13 = tf.nn.elu(bn(conv2d(h12,filters=64,kernel_size=3),is_train))
        h14 = tf.nn.elu(bn(conv2d(h13,filters=128,kernel_size=3),is_train))
        h15 = tf.nn.elu(bn(conv2d(h14,filters=128,kernel_size=3,strides=[2,2]),is_train))
        h16 = tf.nn.elu(bn(conv2d(h15,filters=128,kernel_size=3),is_train))
        h17 = tf.nn.elu(bn(conv2d(h16,filters=96,kernel_size=3),is_train))
        h18 = tf.nn.elu(bn(conv2d(h17,filters=192,kernel_size=3,strides=[2,2]),is_train))
        h19 = tf.nn.elu(bn(conv2d(h18,filters=128,kernel_size=3),is_train))
        h110 = tf.nn.elu(bn(conv2d(h19,filters=160,kernel_size=3),is_train))
        avg_pool1 = tf.reshape(tf.layers.average_pooling2d(h110,pool_size=7,strides=7),[-1,160])
        gen_middle = tf.concat([avg_pool1,z,pose],1)
        fc1 = tf.nn.elu(bn(tf.layers.dense(gen_middle,units=9408,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)),is_train))
        g10 = tf.nn.elu(bn(deconv2d(tf.reshape(fc1,[-1,7,7,192]),
            filters=128,kernel_size=3),is_train))
        g11 = tf.nn.elu(bn(deconv2d(g10,filters=192,kernel_size=3),is_train))
        g12 = tf.nn.elu(bn(deconv2d(g11,filters=192,kernel_size=3,
            strides=[2,2],padding='valid'),is_train))
        g13 = tf.nn.elu(bn(deconv2d(g12,filters=96,kernel_size=3),is_train))
        g14 = tf.nn.elu(bn(deconv2d(g13,filters=128,kernel_size=3),is_train))
        g15 = tf.nn.elu(bn(deconv2d(g14,filters=128,kernel_size=3,strides=[2,2]),is_train))
        g16 = tf.nn.elu(bn(deconv2d(g15,filters=64,kernel_size=3),is_train))
        g17 = tf.nn.elu(bn(deconv2d(g16,filters=64,kernel_size=3),is_train))
        g18 = tf.nn.elu(bn(deconv2d(g17,filters=64,kernel_size=3,strides=[2,2]),is_train))
        g19 = tf.nn.elu(bn(deconv2d(g18,filters=32,kernel_size=3),is_train))
        g110 = deconv2d(g19,filters=1,kernel_size=3)
        generate = tf.nn.tanh(g110)
        return generate
            
def discriminator(image,is_train, reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse) as scope:
        h10 = tf.nn.elu(bn(conv2d(image,filters=32,kernel_size=3),is_train))
        h11 = tf.nn.elu(bn(conv2d(h10,filters=64,kernel_size=3),is_train))
        h12 = tf.nn.elu(bn(conv2d(h11,filters=64,kernel_size=3,strides=[2,2]),is_train))      
        h13 = tf.nn.elu(bn(conv2d(h12,filters=64,kernel_size=3),is_train))
        h14 = tf.nn.elu(bn(conv2d(h13,filters=128,kernel_size=3),is_train))
        h15 = tf.nn.elu(bn(conv2d(h14,filters=128,kernel_size=3,strides=[2,2]),is_train))
        h16 = tf.nn.elu(bn(conv2d(h15,filters=128,kernel_size=3),is_train))
        h17 = tf.nn.elu(bn(conv2d(h16,filters=96,kernel_size=3),is_train))
        h18 = tf.nn.elu(bn(conv2d(h17,filters=192,kernel_size=3,strides=[2,2]),is_train))
        h19 = tf.nn.elu(bn(conv2d(h18,filters=128,kernel_size=3),is_train))
        h110 = tf.nn.elu(bn(conv2d(h19,filters=160,kernel_size=3),is_train))
        avg_pool1 = tf.layers.average_pooling2d(h110,pool_size=7,strides=7)
        avg_pool1_flat = tf.reshape(avg_pool1,[batch_size,160])
        logits_id = tf.layers.dense(avg_pool1_flat,
            units=100,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        logits_pose = tf.layers.dense(avg_pool1_flat,
            units=7,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        real = tf.nn.sigmoid(tf.layers.dense(avg_pool1_flat,
            units=1,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)))
        return logits_id,logits_pose,real
#
G = generator(z,inputs,gen_pose_onehot,is_train=True)
D_id,D_pose,D_real = discriminator(inputs,is_train=True)
D_G_id,D_G_pose,D_G_real = discriminator(G,reuse=True,is_train=True)
#Loss function set
d_loss_id = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y,logits=D_id))
d_loss_pose = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=dis_pose,logits=D_pose))
d_loss_gan = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_G_real))
d_loss = d_loss_pose+d_loss_id+d_loss_gan
g_loss_id = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y,logits=D_G_id))
g_loss_pose = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=gen_pose,logits=D_G_pose))
g_loss_gan = -tf.reduce_mean(tf.log(D_G_real))
g_loss = g_loss_gan+g_loss_pose+g_loss_id
#Variable set
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'D' in var.name]
g_vars = [var for var in t_vars if 'G' in var.name]
#Optimizer set
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01,global_step,
    decay_steps=10,decay_rate=0.99,staircase=True)
d_optim = tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars,
    global_step=global_step)
g_optim = tf.train.AdamOptimizer(learning_rate=0.01).minimize(g_loss,
    var_list=g_vars,global_step=global_step)
#Accuracy ops
d_pose_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(D_pose,1), dis_pose),'int32'))
d_id_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(D_id,1), y),'int32'))
g_pose_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(D_G_pose,1), gen_pose),'int32'))
g_id_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(D_G_id,1), y),'int32'))
#Train
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))  
sess.run(tf.global_variables_initializer())
loss = 0
saver=tf.train.Saver(max_to_keep=1)
for i in range(epoch):
    for j in range(980):
        batch_z = np.random.uniform(-1, 1, [batch_size, 50]).astype(np.float32)
        batch_images,batch_labels,batch_dis_pose,batch_gen_pose=get_batch(j)
        _ = sess.run([d_optim,g_optim],
                feed_dict={ 
                  inputs: batch_images,
                  z: batch_z,
                  y:batch_labels,
                  dis_pose:batch_dis_pose,
                  gen_pose:batch_gen_pose
                })
        _ = sess.run([g_optim],
                feed_dict={
                  inputs: batch_images,
                  z: batch_z,
                  y:batch_labels,
                  dis_pose:batch_dis_pose,
                  gen_pose:batch_gen_pose
                })
        loss_pre = [d_loss_pose,d_loss_id,d_loss_gan,g_loss_gan,g_loss_pose,g_loss_id,
            d_pose_count,d_id_count,g_pose_count,g_id_count]
        dl1,dl2,dl3,gl1,gl2,gl3,d_p,d_id,g_p,g_id = sess.run(loss_pre,
                feed_dict={ 
                  inputs: batch_images,
                  z: batch_z,
                  y:batch_labels,
                  dis_pose:batch_dis_pose,
                  gen_pose:batch_gen_pose
                })
        lr = sess.run(learning_rate)
        saver.save(sess,'ckpt/model.ckpt',global_step=i*980+j)
        print('Epoch - Batch:%d - %d Learning Rate:%f' % (i,j,lr))
        print('D : pose=%f , id=%f, gan=%f  G :pose=%f , id=%f, gan=%f' \
            % (dl1,dl2,dl3,gl1,gl2,gl3))
        print('Acc. real pose:%d / 100, real id:%d / 100, fake pose: %d / 100, \
            fake id:%d / 100' % (d_p,d_id,g_p,g_id))
sess.close()





