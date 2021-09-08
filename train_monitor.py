import numpy as np
import matplotlib.pyplot as plt


    



def latet_parameters(n_examples):
    """

    Parameters
    ----------
    n_examples : TYPE
        DESCRIPTION.

    Returns
    -------
    lat_param : TYPE
        DESCRIPTION.

    """
    lat_param = np.zeros((n_examples,5))
    lat_param[:,0] = (np.random.rand(1,n_examples)*0.4 +0.1).round(4)
    lat_param[:,1] = (np.random.rand(1,n_examples)*0.04 +0.03).round(4)
    lat_param[:,2] = (np.random.rand(1,n_examples)*0.4 +0.5).round(4)
    lat_param[:,3] = (np.random.rand(1,n_examples)*0.4 +0.8).round(4)
    lat_param[:,4] = (np.random.rand(1,n_examples)*0.4 +0.6).round(4)
    return lat_param

def generate_and_save_images(model, epoch, batch_num, test_input,comparison):
    """
    Parameters
    ----------
    model : generator
    epoch : epoch number for the file name
    batch_num : batch number for the file name
    test_input : the seed for the noise dimension of the generator. It's nice 
            if this is kept constant to see how the GAN is evolving through 
            training
    comparison : comparison image from the dataset

    Returns
    -------
    None.
    It saves thge images to directories

    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    lat_param = latet_parameters(3)
    
    predictions = model(test_input, lat_param, training=False)
      
    fig = plt.figure(figsize=(4, 4))
    
    # plot generated images
    for i in range(0,3):
          plt.subplot(2, 2, i+1)
          plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
          plt.axis('off')
    
    # plot real images for comparison
    plt.subplot(2, 2, 4)
    plt.imshow(comparison)
    plt.axis('off')
    plt.savefig('img/image_at_epoch_{}_step_{}.png'.format(epoch,batch_num))
    
    plt.show()
    plt.close("all")
