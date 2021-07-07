import tensorflow as tf
import time
from train_monitor import latet_parameters, generate_and_save_images
import numpy as np


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, epoch,BATCH_SIZE, noise_dim,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer):
    """ 
    This is the fundemental training step for a generatove advisarial network
    decorated with an @tf.function to compile the function for parrallel 
    processing
    """
    
    # define noise vector to be fed into generator
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    lat_params = latet_parameters(BATCH_SIZE)
    
    
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      
        # generate fake images from latent vector and random cosmological
        # parameters in the same range as that of the simulation data
        generated_images = generator(noise, lat_params, training=True)
        
        # get discriminator predictions on both the fake images and a batch
        # of real images from the dataset
        fake_output = discriminator((generated_images, lat_params), training=True)
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

def train(dataset, epochs,BATCH_SIZE, noise_dim,generator,discriminator,
          generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer,
          seed,comparison,checkpoint,checkpoint_prefix, img_save_freq, ckpt_save_freq):
    
    """
    This is the high level training function and loops the training step 
    (defined above). It saves images to a directory every img_save_freq and 
    saves checkpoints every ckpt_save_freq. Note save freq are in batches not
    epochs! This is because the training datasets can be enormous and epochs 
    too long.
    
    seed is a random distribution dimension [num_examples_to_generate, noise_dim]
    """
    for epoch in range(epochs):
        #start timer for each epoch
        
        start1 = time.time()
        i =0
        # do one training step for each image batch
        for batch in dataset:
          
            disc_loss, gen_loss, real_loss, fake_loss = train_step(batch, 
                            epoch,BATCH_SIZE, noise_dim,generator,discriminator,
                            generator_loss,discriminator_loss,generator_optimizer,
                            discriminator_optimizer)
            i+=1
            
            if i % img_save_freq ==0:
                generate_and_save_images(generator,epoch,i,seed,comparison)
                
                print(f"batch: {i}")
                
                # print the losses for the final image batch in this epoch
                print(f"d total: {disc_loss.numpy()}")
                print(f"d real: {real_loss.numpy()}")
                print(f"d fake: {fake_loss.numpy()}")
                print(f"g loss: {gen_loss.numpy()}")
                
            
                # collect and save losses for future reference
                losses = disc_loss, gen_loss, real_loss, fake_loss
                np.save("losses/loss at end of epoch_"+str(epoch+1)+"_step_"+str(i),losses)
            if i % ckpt_save_freq == 0:
                print("saving checkpoint")
                checkpoint.save(file_prefix = checkpoint_prefix)
                print("checkpoint saved")
                
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


