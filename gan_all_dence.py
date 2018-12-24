from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import cv2
import os

import numpy as np

class MOVIE_GAN():
    def __init__(self):
        #Input shape
        self.coordinates = 2
        self.annotations = 18
        self.flames = 32
        self.pose_movie_shape = (self.coordinates, self.annotations, self.flames)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(poses)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(32 * 18 * 2, activation = "relu", input_dim=self.latent_dim))
        model.add(Reshape((2 * 18 * 32)))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        pose_movie = model(noise)

        return Model(noise, pose_movie)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(1, activation='sigmoid', input_shape=self.pose_movie_shape))

        model.summary()

        pose_movie = Input(shape=self.pose_movie_shape)
        validity = model(pose_movie)

        return Model(pose_movie, validity)

    def train(self, epochs, batch_size=32, save_interval=50):

 if __name__ == '__main__':
     movie_gan = MOVIE_GAN()
 movie_gan.train(epochs=500, batch_size=32, save_interval=50)