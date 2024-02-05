import tensorflow as tf
from zipfile import ZipFile
import numpy as np
import PIL
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


BATCH_SIZE = 128
WIDTH = 32
HEIGHT = WIDTH

class Augment(tf.keras.layers.Layer):
  def __init__(self, width, height, seed=42):
    super().__init__()
    
    self.flip_horizontal = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.flip_vertical = tf.keras.layers.RandomFlip(mode="vertical", seed=seed)
    
    self.rotate = tf.keras.layers.RandomRotation( .3 )

    self.resize = tf.keras.layers.Resizing( width=width, 
                                            height=height, 
                                            crop_to_aspect_ratio=True )
    
    self.zoom = tf.keras.layers.RandomZoom( height_factor=(.5, .5) )

  def call( self, x ):
    
    x = self.resize( x )
    
    x = self.flip_vertical( x )
    x = self.flip_horizontal( x )
   
    x = self.rotate( x )
    x = self.zoom( x )

    # x = tf.image.rgb_to_grayscale( x )

    return x / 255.

class PrepareDS( tf.Module ):
    def __init__( self, path, width, height ):
        
        self.zip = ZipFile( path )
        self.info = self.zip.infolist()
        self.augment = Augment( width=WIDTH,
                                height=HEIGHT )

    def __call__( self ):

        for ind, path in enumerate(self.info[ :-10 ]):
            img = self.zip.open( path )
            
            I = np.asarray( PIL.Image.open( img ).convert("RGB") )
            w, h, c = np.shape( I )

            I = tf.cast( self.augment( I ), tf.float32 )
            yield I

prepare_ds = PrepareDS( path="/mnt/c/Users/milos/Downloads/archive.zip",
                        width=WIDTH,
                        height=HEIGHT )

ds = tf.data.Dataset.from_generator( 
    prepare_ds,
    output_signature=tf.TensorSpec( shape=[WIDTH, HEIGHT, 3], dtype=tf.float32 )
).shuffle( 1000 ).batch( BATCH_SIZE )

for example_batch_train in ds.take( 1 ):
#    pass
    for ind, im in enumerate( example_batch_train ):
        plt.subplot( 16, 16, ind + 1 )
        plt.axis( "off" )
        plt.imshow( tf.cast(im * 255., tf.int32), cmap="viridis" )
# plt.show()

class VectorQuantizer( tf.keras.layers.Layer ):
    def __init__( self, num_embeddings, embedding_dim, beta=0.25, **kwargs ):
        super().__init__( **kwargs )
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype=tf.float32
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call( self, x ):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape( x )
        flattened = tf.reshape( x, [-1, self.embedding_dim] )

        # Quantization.
        encoding_indices = self.get_code_indices( flattened )
        encodings = tf.one_hot( encoding_indices, self.num_embeddings )
        quantized = tf.matmul( encodings, self.embeddings, transpose_b=True )

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape( quantized, input_shape )

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
class BaseLayer( tf.keras.layers.Layer ):
    def __init__( self, dropout=0.1, **kwargs ):
        super().__init__()

        self.dropout = tf.keras.layers.Dropout( dropout )
        self.norm = tf.keras.layers.Normalization()
        self.batch_norm = tf.keras.layers.BatchNormalization()

class ResidualBlock( BaseLayer ):
    def __init__( self, filters, kernel_size, **kwargs ):
        super( ResidualBlock, self ).__init__( **kwargs )

        self.conv = tf.keras.layers.Conv2D( filters=filters,
                                            kernel_size=kernel_size, 
                                            padding='same' )

        self.conv1x1 = tf.keras.layers.Conv2D( filters=filters,
                                               kernel_size=1, 
                                               padding='same' )
        
        # self.norm = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.ReLU()

    def call( self, x ):
        x = self.relu( x )
        fx = self.conv( x )

        fx = self.batch_norm( fx )
        # fx = self.dropout( fx )
        
        out = x + fx
        out = self.relu( out )
        
        out = self.batch_norm( out )

        out = self.conv( out )
        
        return self.batch_norm( out )

class Encoder( BaseLayer ):
    def __init__( self, latent_dim, hidden_units, **kwargs ):
        super( Encoder, self ).__init__( **kwargs )

        # self.norm = tf.keras.layers.Normalization()
        # self.norm_2 = tf.keras.layers.Normalization()

        self.conv = [ tf.keras.Sequential([
                      tf.keras.layers.Conv2D( filters=i, 
                                              kernel_size=4,
                                              strides=2, 
                                              padding="same" ),
                      tf.keras.layers.BatchNormalization() ]) for i in [ hidden_units, hidden_units // 2, hidden_units // 4 ] ]

        self.latent_conv = tf.keras.layers.Conv2D( filters=hidden_units,
                                                   kernel_size=1,
                                                   padding="same" )
        
        self.res_block = ResidualBlock( filters=hidden_units, 
                                        kernel_size=3 )
        
    def call( self, x ):
        for conv in self.conv:
            x = conv( x )
            # x = self.norm( x )
        # x = self.dropout( x )
        
        x = self.latent_conv( x )
        x = self.norm( x )

        x = self.res_block( x )
        x = self.res_block( x )
        
        # x = self.dropout( x )

        return x

class Decoder( BaseLayer ):
    def __init__( self, latent_dim, hidden_units, **kwargs ):
        super( Decoder, self ).__init__( **kwargs )

        self.conv = [ tf.keras.Sequential([
                      tf.keras.layers.Conv2DTranspose( filters=i, 
                                                       kernel_size=3,
                                                       strides=2, 
                                                       padding="same" ),
                      tf.keras.layers.BatchNormalization() ]) for i in [ hidden_units // 4, hidden_units // 2, hidden_units ] ]

        self.latent_conv = tf.keras.layers.Conv2DTranspose( filters=3,
                                                            kernel_size=3,
                                                            padding="same" )
        
        self.res_block  = ResidualBlock( filters=hidden_units, 
                                         kernel_size=3 )

    def call( self, x ):
        
        x = self.res_block( x )
        x = self.res_block( x )

        for conv in self.conv:
            x = conv( x )
            # x = self.norm( x )

        x = self.latent_conv( x )
        x = self.batch_norm( x )

        return x

mse = tf.keras.losses.MeanSquaredError()
scce = tf.keras.losses.CategoricalCrossentropy()

class VQVAE( tf.keras.Model ):
    def __init__( self, latent_dim, num_embeddings, hidden_units, beta=0.25, dropout=0.1 ):
        super().__init__()

        self.vq_layer = VectorQuantizer( num_embeddings=num_embeddings,
                                         embedding_dim=latent_dim,
                                         beta=beta,
                                         name="vector_quantizer" )
        
        self.encoder = Encoder( latent_dim=latent_dim, 
                                hidden_units=hidden_units,
                                dropout=dropout )
        
        self.decoder = Decoder( latent_dim=latent_dim, 
                                hidden_units=hidden_units,
                                dropout=dropout )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean( name="reconstruction_loss" )
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

        self.dropout = tf.keras.layers.Dropout( dropout )

    def call( self, x ):

        x = self.encoder( x )
        # x = self.dropout( x )
        
        x = self.vq_layer( x )
        # x = self.dropout( x )

        x = self.decoder( x )
        # x = self.dropout( x )

        return x
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    @tf.function
    def train_step( self, x ):
        with tf.GradientTape() as tape:
            reconstructions = self( x )

            # Calculate the losses.
            # reconstruction_loss = (
            #     tf.reduce_mean((x - reconstructions) ** 2) / self.variance
            # )
            reconstruction_loss = (
                mse(x, reconstructions)
            )
            total_loss = reconstruction_loss + sum( self.losses )

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum( self.losses ))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
# 0.3076
vqvae = VQVAE( beta=.25, 
            #    latent_dim=512, 
               latent_dim=16, 
               num_embeddings=512,
               hidden_units=128,
               dropout=0.3 )

# learning_rate = CustomSchedule( d_model=512 )

optimizers=tf.keras.optimizers.Adam( learning_rate=2e-6 )
vqvae.compile( optimizer=optimizers,
               loss=tf.keras.metrics.SparseCategoricalCrossentropy() )

save_path = "./hrlr_tf_vqvae/vqvae"

vqvae.load_weights( save_path )

# try:
#     tf.saved_model.load( save_path )
# except:
#     print( "failed to load model" )

# 0.0115
vqvae.fit( ds.repeat(),
           batch_size=BATCH_SIZE,
           epochs=1,
           steps_per_epoch=1000 )

vqvae.save_weights( save_path )

for example_batch_train in ds.take( 1 ):
#    pass
    im = example_batch_train[0]

    plt.subplot( 1, 2, 1 )
    plt.axis( "off" )
    plt.title("image")
    plt.imshow( tf.cast(im * 255., tf.int32), cmap="viridis" )

    v = vqvae( im[ tf.newaxis, ... ] )
    v = tf.squeeze( v, axis=0 )

    plt.subplot( 1, 2, 2 )
    plt.axis( "off" )
    plt.title("reconstructed")
    plt.imshow( tf.cast(v * 255., tf.int32), cmap="viridis" )
    # for ind, im in enumerate( example_batch_train[0] ):
plt.show()

# zip = ZipFile( "/mnt/c/Users/milos/Downloads/archive.zip" )

# for ind, path in enumerate( zip.infolist()[60:62] ):
    
#     img = zip.open( path )
        
#     I = np.asarray( PIL.Image.open( img ).convert("RGB") )
#     I = Augment( width=512, height=512 )( I )
#     rec = tf.cast( I[ tf.newaxis, ... ], tf.float32 )
#     v = vqvae( rec )
#     print( v.shape )

#     plt.subplot( 1, 2, 1 )
#     plt.axis( "off" )
#     plt.imshow( tf.cast(I, np.uint8), cmap="viridis" )
    
#     plt.axis( "off" )
#     plt.subplot( 1, 2, 2 )
#     plt.imshow( tf.squeeze(tf.cast(v * 255., np.uint8), axis=0), cmap="viridis" )
# plt.show()

tf.saved_model.save( vqvae, export_dir=save_path )
