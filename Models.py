import tensorflow as tf
from tensorflow.keras import layers


class condition_batch_norm(tf.keras.Model):
    """
    This batch normalisation differs from others by the fact that the 
    conditional labels for this dataset are continous therefore the MLP and 
    architecture are new (although there are simularities).
    
    Note the "channels" needs to be defined when initialising the class. This 
    is dependent on the number of input layers.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.decay = 0.5
        self.epsilon = 1e-05
        self.test_mean = tf.Variable(tf.zeros([self.channels]), dtype=tf.float32, trainable=False)
        self.test_var = tf.Variable(tf.ones([self.channels]), dtype=tf.float32, trainable=False)
        self.beta0 = tf.keras.layers.Dense(units=self.channels, use_bias=True)#, kernel_initializer=weight_init, kernel_regularizer= weight_regularizer_fully)
        self.gamma0 = tf.keras.layers.Dense(units=self.channels, use_bias=True)#, kernel_initializer=weight_init, kernel_regularizer= weight_regularizer_fully)
        self.concat     = layers.Concatenate()
        
        
    def __call__(self, x, training):
        
        x,z,lat = x
        
        invec = self.concat([z,lat])
        
        beta0 = self.beta0(invec)

        gamma0 = self.gamma0(invec)
        
        beta = tf.reshape(beta0, shape=[-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma0, shape=[-1, 1, 1, self.channels])
        
        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            self.test_mean.assign(self.test_mean * self.decay + batch_mean * (1 - self.decay))
            self.test_var.assign(self.test_var * self.decay + batch_var * (1 - self.decay))

            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, self.epsilon)
        else:
            return tf.nn.batch_normalization(x, self.test_mean, self.test_var, beta, gamma, self.epsilon)

class Generator(layers.Layer):
    """ 
    Generator is defined using the call method. Defined the generator and 
    discriminator from scratch to give maximum flexability. Can't use the 
    in built .fit() or .train() if doing this but I have defined the training 
    procedure from scratch anyway in train.py. This is because it gives a lot 
    of flexability when experioemnting with novel training procedures.]
    """
    
    def __init__(self,RES,name="generator",**kwargs):
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
        self.condb1 = condition_batch_norm(512)
        self.condb2 = condition_batch_norm(256)
        self.condb3 = condition_batch_norm(128)
        self.condb4 = condition_batch_norm(64)
    
    
        
    def call(self, lat, param, training):
        
        
        g = self.concat([lat,param])
        g = self.Den1(g)
        g = self.Rel(g)
        g = self.Den2(g)
        g = self.Rel(g)
        g = self.Den3(g)
        g = self.Rel(g)
        g = self.Batch0(g)
        g = self.Reshape(g)
        g = self.Rel(g)
        g = self.condb1([g,param,lat],training)
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
    
    """ 
    Discriminator is defined using the call method. Defined the generator and 
    discriminator from scratch to give maximum flexability. Can't use the 
    in built .fit() or .train() if doing this but I have defined the training 
    procedure from scratch anyway in train.py. This is because it gives a lot 
    of flexability when experioemnting with novel training procedures.]
    """
    
    def __init__(self,RES,name="discriminator",**kwargs):
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
        
        img,param = img
        
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
        d = self.concat([d,param])
        d = self.den1(d)
        d = self.LeRel(d)
        d = self.den2(d)
        d = self.LeRel(d)
        d = self.den3(d)
        d = self.LeRel(d)
        d = self.den4(d)
        
        
        return d