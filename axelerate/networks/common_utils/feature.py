import keras
from keras.models import Model
import tensorflow as tf
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate


from .mobilenet_sipeed.mobilenet import MobileNet

def create_feature_extractor(architecture, input_size, weights = None):
    """
    # Args
        architecture : str
        input_size : int

    # Returns
        feature_extractor : BaseFeatureExtractor instance
    """
    if architecture == 'MobileNet1_0':
        feature_extractor = MobileNetFeature(input_size, weights, alpha=1)
    elif architecture == 'MobileNet7_5':
        feature_extractor = MobileNetFeature(input_size, weights, alpha=0.75)
    elif architecture == 'MobileNet5_0':
        feature_extractor = MobileNetFeature(input_size, weights, alpha=0.5)
    elif architecture == 'MobileNet2_5':
        feature_extractor = MobileNetFeature(input_size, weights, alpha=0.25)
    elif architecture == 'Tiny Yolo':
        feature_extractor = TinyYoloFeature(input_size, weights)
    elif architecture == 'self_dev':
        try:
            feature_extractor = SelfExtractor(input_size, weights)
        except:
            print("failed instantiating a self developed network")
    elif architecture == 'Reduced Tiny Yolo':
        try:
            feature_extractor = ReducedTinyYoloFeature(input_size, weights)
        except:
            print("failed while instantiating reduced tiny yolo!")
    else:
        raise Exception('Architecture not supported! K210 only supports small networks. It should be Tiny Yolo, MobileNet7_5, MobileNet5_0, MobileNet2_5')
    return feature_extractor



class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")       

    def get_input_size(self):
        input_shape = self.feature_extractor.get_input_shape_at(0)
        assert input_shape[1] == input_shape[2]
        return input_shape[1]

    def get_output_size(self):
        output_shape = self.feature_extractor.get_output_shape_at(-1)
        return output_shape[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

class TinyYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights):
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(24*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0,2):
            x = Conv2D(312, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)

        if weights == 'imagenet':
            print('Imagenet for YOLO backend are not available yet, defaulting to random weights')
        elif weights == None:
            pass
        else:
            print('Loaded backend weigths: '+weights)
            self.feature_extractor.load_weights(weights)
    def normalize(self, image):
        return image / 255.

class SelfExtractor (BaseFeatureExtractor):
    #in this class, we define a self built nn
    #standard tiny yolo has 9 convolutional layers
    #in this class we try to reduce the layers of tiny yolo to see if we can improve its performances
    def __init__(self, input_size, weights):
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(24, (3,3), strides=(1,1), padding='same', name='conv_' + str(2), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(1+2))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(48, (3,3), strides=(1,1), padding='same', name='conv_' + str(3), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(1+2))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_' + str(5), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(7))(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        x = Conv2D(312, (3,3), strides=(1,1), padding='same', name='conv_' + str(7), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(7))(x)
        x = LeakyReLU(alpha=0.1)(x)
        print("now building the model: Reduced Tiny Yolo...")


        print("now building the model: Reduced Tiny Yolo...")

        self.feature_extractor = Model(input_image, x)

        if weights == 'imagenet':
            print('Imagenet for YOLO backend are not available yet, defaulting to random weights')
        elif weights == None:
            pass
        else:
            print('Loaded backend weigths: '+weights)
            self.feature_extractor.load_weights(weights)
    def normalize(self, image):
        return image / 255.

class ReducedTinyYoloFeature(BaseFeatureExtractor):
    #standard tiny yolo has 9 convolutional layers
    #in this class we try to reduce the layers of tiny yolo to see if we can improve its performances
    def __init__(self, input_size, weights):
        input_image = Input(shape=(input_size[0], input_size[1], 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)


        x = Conv2D(24, (3,3), strides=(1,1), padding='same', name='conv_' + str(2), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(2))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(48, (3,3), strides=(1,1), padding='same', name='conv_' + str(1+2), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(1+2))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        x = Conv2D(312, (3,3), strides=(1,1), padding='same', name='conv_' + str(7), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(7))(x)
        x = LeakyReLU(alpha=0.1)(x)
        print("now building the model: Reduced Tiny Yolo...")
        self.feature_extractor = Model(input_image, x)

        if weights == 'imagenet':
            print('Imagenet for YOLO backend are not available yet, defaulting to random weights')
        elif weights == None:
            pass
        else:
            print('Loaded backend weigths: '+weights)
            self.feature_extractor.load_weights(weights)
    def normalize(self, image):
        return image / 255.


class MobileNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size, weights, alpha):
        input_image = Input(shape=(input_size[0], input_size[1], 3))
        input_shapes_imagenet = [(128, 128,3), (160, 160,3), (192, 192,3), (224, 224,3)]
        input_shape =(128,128,3)
        for item in input_shapes_imagenet:
            if item[0] <= input_size[0]:
                input_shape = item

        if weights == 'imagenet':
            mobilenet = MobileNet(input_shape=input_shape, input_tensor=input_image, alpha = alpha, weights = 'imagenet', include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
            print('Successfully loaded imagenet backend weights')
        else:
            mobilenet = MobileNet(input_shape=(input_size[0],input_size[1],3),alpha = alpha,depth_multiplier = 1, dropout = 0.001, weights = None, include_top=False, backend=keras.backend, layers=keras.layers,models=keras.models,utils=keras.utils)
            if weights:
                print('Loaded backend weigths: '+weights)
                mobilenet.load_weights(weights)

        #x = mobilenet(input_image)
        self.feature_extractor = mobilenet

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image		
