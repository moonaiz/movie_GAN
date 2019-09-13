from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tqdm import tqdm

import cv2
import os
import pandas as pd
import json
import csv
import keras.backend as K

import numpy as np

from keras.engine.topology import Network

OLD2NEW = [[2,0], [3,1], [4,2], [5,3], [6,4], [7,5], [8,6], [9,7],
            [10,8], [11,9], [12,10], [13,11], [1,15], [14,13], [15,14],
            [0,12], [16,16], [17,17]]

class MOVIE_GAN():
    def __init__(self):
        #Input shape
        self.coordinates = 2
        self.annotations = 18
        self.flames = 32
        self.pose_movie_shape = (self.flames, self.annotations, self.coordinates)
        self.latent_dim = 100
        self.msgan_parameter = 0.1

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        self.generator2 = Network(inputs=self.generator.inputs, outputs=self.generator.outputs)

        # The generator takes noise as input and generates imgs
        z1 = Input(shape=(self.latent_dim,))
        z2 = Input(shape=(self.latent_dim,))
        #gen1 = self.generator(z[0])
        poses1 = self.generator(z1)
        poses2 = self.generator2(z2)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(poses1)

        #new_pred = valid + self.msgan_parameter * (self.compute_poses_distance(gen1,gen2)/abs(z1-z2))

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator

        loss = Lambda(self.msgan_loss, output_shape = (1,), name = "msgan_loss")([z1,z2,poses1,poses2,valid])
        self.combined = Model([z1,z2],loss)

        self.combined.compile(loss={"msgan_loss" : lambda y_true, y_pred : y_pred},
            optimizer=optimizer,
            metrics=['accuracy'])

    def msgan_loss(self, args):

        '''
        z1 = args[0]
        z2 = args[1]
        poses1 = args[2]
        poses2 = args[3]
        valid = args[4]

        dist = self.compute_poses_distance(poses1, poses2) / np.linalg.norm(z1-z2)

        loss = self.msgan_parameter * dist
        loss += valid
        '''

        valid = args[4]
        y_true = K.ones_like(valid)

        return K.binary_crossentropy(valid, y_true)


    def compute_poses_distance(self, gen1, gen2):

        dist = 0

        for i in range(self.flames):
            for j in range(self.annotations):
                for k in range(self.coordinates):
                    dist += abs(gen1[i,j,k] - gen2[i,j,k])

        return dist

    def build_generator(self):
        model = Sequential()

        model.add(Dense(32 * 128 * 64, activation = "tanh", input_dim=self.latent_dim))
        model.add(Reshape((32, 128 ,64)))
        model.add(Conv2D(64, kernel_size=(3, 4), strides=(1, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv2D(32, kernel_size=(3, 4), strides=(1, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv2D(16, kernel_size=(3, 4), strides=(1, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(32 * 18 * 2, activation="tanh"))
        model.add(Reshape((32, 18, 2)))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        pose_movie = model(noise)

        return Model(noise, pose_movie)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=(3, 4), strides=(1, 2), padding='same', input_shape=self.pose_movie_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        pose_movie = Input(shape=self.pose_movie_shape)
        validity = model(pose_movie)

        return Model(pose_movie, validity)

    def load_pose_cords(self, y_str, x_str):
        y_cords = json.loads(y_str)
        x_cords = json.loads(x_str)
        cords = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
        return cords.astype(np.int)

    def train(self, epochs, batch_size=32, save_interval=50):
        #from pose_utils import load_pose_cords_from_string
        input_folder = './annotations/walk2/'
        annotation_list = os.listdir(input_folder)

        train = np.zeros((len(annotation_list), ) + self.pose_movie_shape)

        for i in range(len(annotation_list)):
            df = pd.read_csv(input_folder + '%s'% annotation_list[i], sep=':')
            df = df.sort_values('name')

            t = 0

            for index, row in df.iterrows():
                train[i][t] = self.load_pose_cords(row['keypoints_y'], row['keypoints_x'])
                t += 1

#train data fitting

        for i in range(len(annotation_list)):
            for j in range(self.flames):
                for k in range(self.annotations):
                    for l in range(self.coordinates):

                        if train[i,j,k,l] == -1:
                            if j == 0:
                                train[i,j,k,l] = (train[i,j+1,k,l] + train[i,j+2,k,l])/2
                            elif not j == self.flames - 1:
                                train[i,j,k,l] = (train[i,j-1,k,l] + train[i,j+1,k,l])/2
                            else:
                                train[i,j,k,l] = (train[i,j-1,k,l] + train[i,j-2,k,l])/2

        train = train / 127.5 - 1

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        f = open('gan_loss.csv','w')
        writer = csv.writer(f)
        writer.writerow(['epoch','D_loss','accuracy','G_loss'])

        for epoch in range(epochs):

            idx = np.random.randint(0, train.shape[0], batch_size)
            pose_movie_train = np.zeros((batch_size, 32, 18, 2), dtype=float)

            for i in range(batch_size):
                pose_movie_train[i] = train[idx[i]]


            noise1 = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise2 = np.random.normal(0, 1, (batch_size, self.latent_dim))

            pose_movie_gen = self.generator.predict(noise1)


            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(pose_movie_train, valid)#valid=real
            d_loss_fake = self.discriminator.train_on_batch(pose_movie_gen, fake)#fake
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([noise1, noise2], valid)

            # Plot the progress

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
            writer.writerow([epoch, d_loss[0], 100*d_loss[1], g_loss[0]])

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_annotations(epoch)

        f.close()

    def save_annotations(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        pose_movie_gen = self.generator.predict(noise)#gen_pose -1 - 1

        if not os.path.exists('./output'):
            os.mkdir('./output')

        output_folder = './output/msgans/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Rescale 0 - 1
        pose_movie_gen = 0.5 * pose_movie_gen + 0.5

        # Rescale 0 - 255
        pose_movie_gen = 255 * pose_movie_gen

        pose_movie_gen = pose_movie_gen.astype('int32')


        for i in range(r * c):
            output_path = output_folder + "epoch_%d-%d.csv" %(epoch, i)
            result_file = open(output_path, 'w')
            processed_names = set()
            print('name:keypoints_y:keypoints_x',file=result_file)
            for t in range(32):
                print('%s.jpg: %s: %s' % ('{0:02d}'.format(t), str(list(pose_movie_gen[i, t, :, 0])), str(list(pose_movie_gen[i, t, :, 1]))), file=result_file)
                result_file.flush()


if __name__ == '__main__':
    movie_gan = MOVIE_GAN()

movie_gan.train(epochs=501, batch_size=32, save_interval=500)
