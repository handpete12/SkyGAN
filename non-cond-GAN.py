#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:45:42 2021

@author: Project
"""


import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time

# Check GPU(s) has been detected and is usable.
print("Num GPUs Available: ", 
      len(tf.config.experimental.list_physical_devices('GPU')))
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], enable=True)
tf.config.experimental.set_memory_growth(devices[1], enable=True)

# define bufer, batch size, resolution, num of epochs, noise dimension (broken)
# and number of examples to generate (not fully operational yet)

BATCH_SIZE = 16 #size of batch
RES = 128 #define resolution of input images (GAN supports square # of 2)
EPOCHS = 200
noise_dim = 100 # broken for nopw, leave as is
num_examples_to_generate = 64 # do not change until fixed generate and save
seed = tf.random.normal([num_examples_to_generate, noise_dim])


try:
    os.mkdir('img')
    print("dir created")
except:
    print("dir exists")
else:
    print("dir exists")
        
try:
    os.mkdir('losses')
    print("dir created")
except:
    print("dir exists")
else:
    print("dir exists")
try:
    os.mkdir('mass')
    print("dir created")
except:
    print("dir exists")
else:
    print("dir exists")

# load the cosmological parameters outside the generator, tensorflow reccomends
# this.
# cosmo = np.load('/home/handpete/Documents/Cosmological_Parameters/'+
#                 'a_All_Cosmological_Parameters.npy')

MASS_AREA = np.load("MASS_AREA.npy")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 16:17:36 2021

@author: handpete
"""


# newpath = r'/home/handpete/Desktop/New Project 14.04.21/Datasets/680000 - normalised params' 
# newpath = r'tf dataset norm params'
# newpath = r'/Users/Project/Desktop/Cosmo_0_tf/'
newpath = r'Cosmo_0_tf'



train_dataset = tf.data.experimental.load(
    newpath, element_spec=(tf.TensorSpec(shape=(128,128,1),dtype='float32'))).batch(BATCH_SIZE)


# params  = train_dataset.map(lambda x, y: y).batch(BATCH_SIZE)
# train_dataset = train_dataset.map(lambda x, y: x).batch(BATCH_SIZE)



for x in train_dataset.take(1):
    comparison=x.numpy()[0,:,:,0]
    plt.imshow(comparison)
print("-------------------------------------------------------------------")
print("initialised dataset")
print("Num GPUs Available: ", 
      len(tf.config.experimental.list_physical_devices('GPU')))
print("-------------------------------------------------------------------")
#%% #######  Define Generator and Discriminator using call method  #######

class Generator(layers.Layer):
    
    def __init__(self,name="generator",**kwargs):
        super().__init__(name=None)
        #self.In_lat    = layers.Input(shape=(None,100))
        #self.In_param  = layers.Input(shape=(None,5))
        self.concat     = layers.Concatenate()
        self.fix        = layers.Dense(100)
        self.Den1       = layers.Dense(256,use_bias=False,name="G_den_1",input_shape=(None, 100))
        self.Den2       = layers.Dense(512,use_bias=False,name="G_den_2",input_shape=(None, 256))
        self.Den3       = layers.Dense(int(RES/(2**4)*RES/(2**4)*512),use_bias=False,name="G_den_3",input_shape=(None, 512))
        self.Rel        = layers.ReLU(0.2)
        self.Batch      = layers.BatchNormalization(momentum=0.5)
        self.Batch0     = layers.BatchNormalization(momentum=0.5,name="GB1")
        self.Batch1     = layers.BatchNormalization(momentum=0.5,name="GB2",input_shape=( int(RES/(2**4)), int(RES/(2**4)), 512))
        self.Batch2     = layers.BatchNormalization(momentum=0.5,name="GB3",input_shape=( int(RES/(2**3)), int(RES/(2**3)), 256))
        self.Batch3     = layers.BatchNormalization(momentum=0.5,name="GB4",input_shape=( int(RES/(2**2)), int(RES/(2**2)), 128))
        self.Batch4     = layers.BatchNormalization(momentum=0.5,name="GB5",input_shape=( int(RES/(2**1)), int(RES/(2**1)), 64))
        self.Reshape    = layers.Reshape((int(RES/(2**4)), int(RES/(2**4)), 512))
        self.ConvT1     = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.ConvT2     = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.ConvT3     = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same',use_bias=False)
        self.ConvT4     = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',use_bias=False, activation='tanh')
        
        # Defining neseccary stuff for conditional batch norm
        
        
    
    
        
    def call(self, lat, training):
        
        
        
        
        # splits = tf.split(lat, 4, axis=1)    
        
        
        
        g = self.Den1(lat)
        g = self.Rel(g)
        g = self.Den2(g)
        g = self.Rel(g)
        g = self.Den3(g)
        g = self.Rel(g)
        g = self.Batch0(g)
        g = self.Reshape(g)
        g = self.Rel(g)
        g = self.Batch1(g)
        g = self.ConvT1(g)
        g = self.Rel(g)
        g = self.Batch2(g)
        g = self.ConvT2(g)
        g = self.Rel(g)
        g = self.Batch3(g)
        g = self.ConvT3(g)
        g = self.Rel(g)
        g = self.Batch4(g)
        g = self.ConvT4(g)
        
        return g


class Discriminator(layers.Layer):
    
    def __init__(self,name="discriminator",**kwargs):
        super().__init__(name=None)
        
        # self.In_param   = tf.keras.Input(shape=(None,5))
        # self.In_]img     = tf.keras.Input(shape=(None,RES,RES,1))
        self.Conv1      = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[int(RES/(2**0)), int(RES/(2**0)), 1])
        self.LeRel      = layers.LeakyReLU(0.2)
        self.Batch4     = layers.BatchNormalization(momentum=0.5,name="DB4",input_shape=( int(RES/(2**4)), int(RES/(2**4)), 512))
        self.Batch3     = layers.BatchNormalization(momentum=0.5,name="DB3",input_shape=( int(RES/(2**3)), int(RES/(2**3)), 256))
        self.Batch2     = layers.BatchNormalization(momentum=0.5,name="DB2",input_shape=( int(RES/(2**2)), int(RES/(2**2)), 128))
        self.Batch1     = layers.BatchNormalization(momentum=0.5,name="DB1",input_shape=( int(RES/(2**1)), int(RES/(2**1)), 64))
        self.Conv2      = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.Conv3      = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')
        self.Conv4      = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')
        self.flat       = layers.Flatten()
        self.den1       = layers.Dense(512,name="dis1")
        self.den2       = layers.Dense(256,name="dis1")
        self.den3       = layers.Dense(128,name="dis1")
        self.den4       = layers.Dense(1,activation='sigmoid',name="dis1")
        self.concat     = layers.Concatenate()
        
        
    def call(self, img):
        
        
        # img = self.In_img(img)
        # l = self.In_param(param)
        d = self.Conv1(img)
        d = self.LeRel(d)
        d = self.Batch1(d)
        d = self.Conv2(d)
        d = self.LeRel(d)
        d = self.Batch2(d)
        d = self.Conv3(d)
        d = self.LeRel(d)
        d = self.Batch3(d)
        d = self.Conv4(d)
        d = self.LeRel(d)
        d = self.Batch4(d)
        d = self.flat(d)
        d = self.den1(d)
        d = self.LeRel(d)
        d = self.den2(d)
        d = self.LeRel(d)
        d = self.den3(d)
        d = self.LeRel(d)
        d = self.den4(d)
        
        
        return d

#%% ####################  Define Loss functions  ##############################

# This method returns a helper function to compute cross entropy loss
# defining the loss function for both models

Loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = Loss_function(tf.ones_like(real_output), real_output)
    fake_loss = Loss_function(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss

def generator_loss(fake_output):
    return Loss_function(tf.ones_like(fake_output), fake_output)


#%% ###############  Define The Training Procedure  ######################


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled". I think this is 
# nessecary for it to run an a gpu will have to ask if that is so
@tf.function
def train_step(images, epoch):
    
    # define noise vector to be fed into generator
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    
    
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      
        # generate fake images from latent vector and random cosmological
        # parameters in the same range as that of the simulation data
        generated_images = generator(noise, training=True)
        
        # get discriminator predictions on both the fake images and a batch
        # of real images from the dataset
        fake_output = discriminator(generated_images, training=True)
        real_output = discriminator(images, training=True)
        
        # calculate the losses based off the chosen loss function for both
        # the discriminator and generator
        gen_loss = generator_loss(fake_output)
        disc_loss, real_loss, fake_loss= discriminator_loss(real_output,
                                                          fake_output)
      
    
    # Calculate the gradient steps for the models from the losses calculated 
    # previously. 
    gradients_of_generator = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                            discriminator.trainable_variables)

    # Apply the gradient steps to the generator and discriminator
    generator_optimizer.apply_gradients(zip(gradients_of_generator, 
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                            discriminator.trainable_variables))
    
    # returning these parameters purly for the loss history 
    return disc_loss, gen_loss, real_loss, fake_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        #start timer for each epoch
        start = time.time()
        start1 = time.time()
        i =0
        # do one training step for each image batch
        for batch in dataset:
          
            disc_loss, gen_loss, real_loss, fake_loss = train_step(batch, epoch)
            i+=1
            
            if i % 300 ==0:
                generate_and_save_images(generator,epoch,i,seed)
                
                print(i)
                print(time.time()-start)
                start=time.time()
                # print the losses for the final image batch in this epoch
                print(str(epoch)+"."+str(i)+"dt: "+str(disc_loss.numpy()))
                print(str(epoch)+"."+str(i)+"dr: "+str(real_loss.numpy()))
                print(str(epoch)+"."+str(i)+"df: "+str(fake_loss.numpy()))
                print(str(epoch)+"."+str(i)+"gl: "+str(gen_loss.numpy()))
            
                # collect and save losses for future reference
                losses = disc_loss, gen_loss, real_loss, fake_loss
                np.save("losses/loss at end of epoch_"+str(epoch+1)+"_step_"+str(i),losses)
            
        checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start1))

#%% #################  Miscelaneous Functions  ############################

def mass_histogram(dens_field,plot):
    
    bins = np.logspace(10,14,60)#np.logspace(12,16,60)
    # plt.hist(data_real,bins=bins,stacked=True,facecolor='blue',histtype='stepfilled')
    a = np.histogram(dens_field,bins=bins, )
    
    if plot==True:  
        plt.plot(bins[1:], a[0])
        plt.xscale("log")
        plt.yscale("linear")
        
    return bins, a

def inverse_smooth(s,a):
    return (s+1)*a/(1-s)

def latet_parameters(n_examples):
    lat_param = np.zeros((n_examples,5))
    lat_param[:,0] = (np.random.rand(1,n_examples)*0.4 +0.1).round(4)
    lat_param[:,1] = (np.random.rand(1,n_examples)*0.04 +0.03).round(4)
    lat_param[:,2] = (np.random.rand(1,n_examples)*0.4 +0.5).round(4)
    lat_param[:,3] = (np.random.rand(1,n_examples)*0.4 +0.8).round(4)
    lat_param[:,4] = (np.random.rand(1,n_examples)*0.4 +0.6).round(4)
    return lat_param

def generate_and_save_images(model, epoch,j, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    
    
    predictions = model(test_input, training=False)
      
    fig = plt.figure(figsize=(10, 10))
    
    # plot generated images
    for i in range(0,3):
          plt.subplot(2, 2, i+1)
          plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
          plt.axis('off')
    
    # plot real images for comparison
    plt.subplot(2, 2, 4)
    plt.imshow(comparison)
    plt.axis('off')
    plt.savefig('img/image_at_epoch_{}_step_{}.png'.format(epoch,j))
    
    #######################################
    fig = plt.figure(figsize=(10, 10))
    
    MASSES_g = np.zeros((num_examples_to_generate,59))
    for i in range(0,num_examples_to_generate):
            
            
            bins,masses_g = mass_histogram(inverse_smooth(predictions[i,:,:,0],2e12),False)
            MASSES_g[i,:] = masses_g[0]
    
    upper_g = np.max(MASSES_g,axis=0)
    lower_g = np.min(MASSES_g,axis=0)
    mean_g = np.mean(MASSES_g,axis=0)
    
    
    
    plt.fill_between(bins[1:],upper_g,lower_g,alpha=1,facecolor='red')
    plt.fill_between(bins[1:],MASS_AREA[0],MASS_AREA[1],alpha=0.6,facecolor='blue')
    plt.plot(bins[1:],masses_g[0],'k-')
    # plt.legend(["generated","real"])
    plt.xscale("log")
    plt.xlim([1e10,10e14])
    plt.yscale("linear")
    plt.show()
    
    plt.savefig('mass/image_at_epoch_{}_step_{}.png'.format(epoch,j))
    plt.close('all')

#%%
generator = Generator()
discriminator = Discriminator()


# The discriminator and the generator optimizers are different since we 
# train two networks separately.
generator_optimizer = tf.keras.optimizers.Adam(lr=1e-5)#for WGAN change to 1e-8
discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.5,
                                                   beta_2=0.999)

# defines checkpoints directory ready for the training function definition
# or ready to restor from last checkpoint if nesessary

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                            discriminator_optimizer=discriminator_optimizer,
                            generator=generator,
                            discriminator=discriminator)


#########       if want to restore from a checkpoint        #############

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#########################################################################

# Train the model
train(train_dataset, EPOCHS)



