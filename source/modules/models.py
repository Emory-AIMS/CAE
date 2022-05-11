from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, AveragePooling2D, MaxPooling2D,BatchNormalization, LeakyReLU
from keras.layers import Flatten,Reshape,Conv2DTranspose,Activation,concatenate,Input, Concatenate, Dropout, Multiply,Add
import keras.regularizers as regs
from config import setup_argparse
parser = setup_argparse()
args = parser.parse_args()


def conditional_cae():

    image_size = args.dim12
    channel = args.dim3

    input_shape = (image_size, image_size, channel)
    input_x = Input(shape=input_shape, name='encoder_input')

    n_channels = input_shape[-1]
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(input_x)
    x = MaxPooling2D((2,2),padding='same')(x)
    half = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(half)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
    x_out = Conv2D(n_channels, (3,3), activation='sigmoid', padding='same')(x)

    y = Dropout(0.25) (half) ###???
    y = Flatten()(half)
    y = Dense(128, activation='sigmoid')(y)
    y = Dropout(0.5)(y)
    y_out = Dense(2, activation='softmax')(y)


    return Model(inputs=input_x,outputs=[x_out,y_out])



def autoencoder():

    image_size = args.dim12
    channel = args.dim3

    input_shape = (image_size, image_size, channel)
    input_x = Input(shape=input_shape, name='encoder_input')

    n_channels = input_shape[-1]
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(input_x)
    x = MaxPooling2D((2,2),padding='same')(x)
    half = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(half)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
    x_out = Conv2D(n_channels, (3,3), activation='sigmoid', padding='same')(x)

    return Model(inputs=input_x,outputs=x_out)


def autoencoder_cifar():
    image_size = args.dim12
    channel = args.dim3

    input_shape = (image_size, image_size, channel)
    input_x = Input(shape=input_shape, name='encoder_input')

    n_channels = input_shape[-1]
    half = Conv2D(3, (3,3), activation='sigmoid', padding='same')(input_x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(half)
    x_out = Conv2D(n_channels, (3,3), activation='sigmoid', padding='same')(x)

    return Model(inputs=input_x,outputs=x_out)


def conditional_cae_cifar():

    ## prev model but in Magnet, BEST MODEL EVER (classfier + AE)
    image_size = args.dim12
    channel = args.dim3

    input_shape = (image_size, image_size, channel)
    input_x = Input(shape=input_shape, name='encoder_input')

    n_channels = input_shape[-1]
    half = Conv2D(3, (3,3), activation='sigmoid', padding='same')(input_x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(half)
    x_out = Conv2D(n_channels, (3,3), activation='sigmoid', padding='same')(x)

    y = Dropout(0.25) (half) ###???
    y = Flatten()(half)
    y = Dense(128, activation='sigmoid')(y)
    # y = Dense(10, activation='sigmoid')(y)
    y = Dropout(0.5)(y)
    y_out = Dense(2, activation='softmax')(y)


    return Model(inputs=input_x,outputs=[x_out,y_out])

def magnet_1():

    image_size = args.dim12
    channel = args.dim3

    input_shape = (image_size, image_size, channel)
    input_x = Input(shape=input_shape, name='encoder_input')

    n_channels = input_shape[-1]
    half = Conv2D(3, (3,3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(input_x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(half)
    x_out = Conv2D(n_channels, (3,3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(x)

    return Model(inputs=input_x,outputs=x_out)

def magnet_2():

    image_size = args.dim12
    channel = args.dim3

    input_shape = (image_size, image_size, channel)
    input_x = Input(shape=input_shape, name='encoder_input')

    n_channels = input_shape[-1]
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(input_x)
    x = AveragePooling2D((2,2),padding='same' )(x)
    half = Conv2D(3, (3,3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(half)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(x)
    x_out = Conv2D(n_channels, (3,3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9))(x)

    return Model(inputs=input_x,outputs=x_out)

def load_model(name):
    print(name)
    if args.dataset:
        if name=='autoencoder':
            return autoencoder_cifar()
        if name=='magnet_1':
            return magnet_1()
        if name=='magnet_2':
            return magnet_1()
        elif name=='conditional_cae':
            return conditional_cae_cifar()
        else:
            raise ValueError('Unknown model name %s was given' % name)
    else:
        if name=='autoencoder':
            return autoencoder()
        if name=='magnet_1':
            return magnet_1()
        if name=='magnet_2':
            return magnet_2()
        elif name=='conditional_cae':
            return conditional_cae()
        else:
            raise ValueError('Unknown model name %s was given' % name)
