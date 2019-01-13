from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from tqdm import tqdm

import keras.backend as K
import cv2
import os
import pandas as pd
import json
import csv

import numpy as np

OLD2NEW = [[2,0], [3,1], [4,2], [5,3], [6,4], [7,5], [8,6], [9,7],
            [10,8], [11,9], [12,10], [13,11], [1,15], [14,13], [15,14],
            [0,12], [16,16], [17,17]]

class MOVIE_GAN():
    def __init__(self):
        #Input shape
        self.coordinates = 2
        self.annotations = 18
        self.flames = 32
        self.num_classes = 3
        self.pose_movie_shape = (self.flames, self.annotations, self.coordinates)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', self.mutual_info_loss]

        # Build and compile the discriminator
        self.discriminator, self.auxilliary = self.build_disk_and_q_net()

        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the recognition network Q
        self.auxilliary.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes gen_input as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        poses = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(poses)

        target_label = self.auxilliary(poses)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)

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

        gen_input = Input(shape=(self.latent_dim,))
        pose_movie = model(gen_input)

        model.summary()

        return Model(gen_input, pose_movie)

    def build_disk_and_q_net(self):

        pose_movie = Input(shape=self.pose_movie_shape)

        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 4), strides=(1, 2), padding='same', input_shape=self.pose_movie_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Flatten())

        pose_embedding = model(pose_movie)

        #Discriminator
        validity = Dense(1, activation='sigmoid')(pose_embedding)

        #Recognition
        q_net = Dense(128, activation='relu')(pose_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)

        # Return discriminator and recognition network
        return Model(pose_movie, validity), Model(pose_movie, label)

    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy

    def load_pose_cords(self, y_str, x_str):
        y_cords = json.loads(y_str)
        x_cords = json.loads(x_str)
        cords = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
        return cords.astype(np.int)

    def replace_index(self, stickman, flag):
        stick_old = stickman
        stick_new = np.zeros((18,2), dtype = float)
        for f, t in OLD2NEW:

            if flag == True:
                stick_new[t][:] = stick_old[f][:]
            else:
                stick_new[f][:] = stick_old[t][:]

        return stick_new

    def sample_generator_input(self, batch_size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, self.latent_dim - self.num_classes))
        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

        return sampled_noise, sampled_labels

    def train(self, epochs, batch_size=32, save_interval=50):
        #from pose_utils import load_pose_cords_from_string
        input_folder = './annotations/'
        annotation_list = os.listdir(input_folder)

        train = np.zeros((len(annotation_list), ) + self.pose_movie_shape)

        for i in range(len(annotation_list)):
            df = pd.read_csv(input_folder + '%s'% annotation_list[i], sep=':')
            df = df.sort_values('name')

            t = 0

            for index, row in df.iterrows():
                train[i][t] = self.load_pose_cords(row['keypoints_y'], row['keypoints_x'])
                t += 1

        train = train / 127.5 - 1

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        f = open('gan_loss.csv','a')
        writer = csv.writer(f)
        writer.writerow(['epoch','D_loss','accuracy','G_loss1','G_loss2'])

        for epoch in range(epochs):

            idx = np.random.randint(0, train.shape[0], batch_size)
            pose_movie_train = np.zeros((batch_size, 32, 18, 2), dtype=float)

            for i in range(batch_size):
                pose_movie_train[i] = train[idx[i]]
                for t in range(32):
                    pose_movie_train[i][t] = self.replace_index(pose_movie_train[i][t], True)

            # Sample gen_input and categorical labels
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)


            pose_movie_gen = self.generator.predict(gen_input)


            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(pose_movie_train, valid)#valid=real
            d_loss_fake = self.discriminator.train_on_batch(pose_movie_gen, fake)#fake
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels])

            # Plot the progress

            print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))
            writer.writerow([epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]])

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_annotations(epoch)

        f.close()

    def save_annotations(self, epoch):

        if not os.path.exists('./output'):
            os.mkdir('./output')

        output_folder = './output/annotations/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        r, c = self.num_classes, self.num_classes

        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(c)
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)

            gen_input = np.concatenate((sampled_noise, label), axis=1)
            pose_movie_gen = self.generator.predict(gen_input)#gen_img -1 - 1

        # Rescale 0 - 1
            pose_movie_gen = 0.5 * pose_movie_gen + 0.5
        # Rescale 0 - 255
            pose_movie_gen = 255 * pose_movie_gen
            pose_movie_gen = pose_movie_gen.astype('int32')

            for j in range(r):
                for t in range(32):
                    pose_movie_gen[j][t] = self.replace_index(pose_movie_gen[j][t], False)

                output_path = output_folder + "epoch_%d(%d-%d).csv" %(epoch, int(i+1), int(j+1))

                result_file = open(output_path, 'w')
                processed_names = set()
                print('name:keypoints_y:keypoints_x',file=result_file)

                for t in range(32):
                    print('%s.jpg: %s: %s' % ('{0:02d}'.format(t), str(list(pose_movie_gen[j, t, :, 0])), str(list(pose_movie_gen[j, t, :, 1]))), file=result_file)

                result_file.flush()


if __name__ == '__main__':
    movie_gan = MOVIE_GAN()

movie_gan.train(epochs=10001, batch_size=32, save_interval=1000)
