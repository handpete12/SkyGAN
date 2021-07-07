#%% 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:05:35 2021

@author: Peter Handy
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os


# define bufer, batch size, resolution, num of epochs
# and number of examples to generate
BATCH_SIZE = 64 
RES = 128 # define resolution of input images (GAN supports square # of 2)
EPOCHS = 25
noise_dim = 1028 # broken for now, leave as is
num_examples_to_generate = 3 # do not change (I'm uploading a fix soon)




# Check GPU(s) has been detected and is usable.
print("Num GPUs Available: ", 
      len(tf.config.experimental.list_physical_devices('GPU')))
devices = tf.config.experimental.list_physical_devices('GPU')
                                                       
# Tensorflow ocasionally has a memory bug with 2080 Ti graphiocs cards. If you 
# experience this bug the use the following line(s) (depending how many GPUs)

# tf.config.experimental.set_memory_growth(devices[0], enable=True)
# tf.config.experimental.set_memory_growth(devices[1], enable=True)





def create_save_directories():

    try:
        os.mkdir('img')
        print("dir created")
    except:
        print("dir already exists")
    
            
    try:
        os.mkdir('losses')
        print("dir created")
    except:
        print("dir already exists")
        
create_save_directories()


newpath = r'Test' # path to the dataset. The dataset in the github only 
# contains one batch and is useful for debugging code. 



# note the dataset has to be saved as  tf.data.experimental.save
# see tensorflow documentation here:
# https://www.tensorflow.org/api_docs/python/tf/data/experimental/save
train_dataset = tf.data.experimental.load(
    newpath, element_spec=(tf.TensorSpec(shape=(128,128,1),dtype='float32'),tf.TensorSpec(
            shape=(5,),dtype='float32')), compression=None, reader_func=None
).batch(BATCH_SIZE).shuffle(1000)


# This is to test that the dataset has loaded propely and displayts the first
# image. It also defines a comparison image that will be plotted next to the 
# GAN genersated image during training. 
for x,y in train_dataset.take(1):
    comparison=x.numpy()[0,:,:,0]
    plt.imshow(comparison)
print("-------------------------------------------------------------------")
print("initialised dataset")
print("Num GPUs Available: ", 
      len(tf.config.experimental.list_physical_devices('GPU')))
print("-------------------------------------------------------------------")
#%% #######  Define Generator and Discriminator using call method  #######

from Models import Generator, Discriminator

#%% ####################  Define Loss functions  ##############################

Loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = Loss_function(tf.ones_like(real_output), real_output)
    fake_loss = Loss_function(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss

def generator_loss(fake_output):
    return Loss_function(tf.ones_like(fake_output), fake_output)


#%% ###############  Define The Training Procedure  ######################

from train import train

#%%
generator = Generator(RES=RES)
discriminator = Discriminator(RES=RES)


# The discriminator and the generator optimizers are different since we can
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
train(train_dataset, EPOCHS,BATCH_SIZE, noise_dim,generator,discriminator,
      generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer,
      seed=tf.random.normal([num_examples_to_generate, noise_dim]),comparison,
      checkpoint,checkpoint_prefix,img_save_freq=1,ckpt_save_freq=6)



