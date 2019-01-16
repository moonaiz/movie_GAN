from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding
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
        self.num_classes = 2
        self.pose_movie_shape = (self.flames, self.annotations, self.coordinates)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        poses = self.generator([z, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator([poses, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, label], valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

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
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        pose_movie = model(model_input)

        return Model([noise, label], pose_movie)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=(3, 4), strides=(1, 2), padding='same', input_shape=(self.flames,self.annotations, self.coordinates + self.num_classes)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        pose_movie = Input(shape=self.pose_movie_shape)
        label = Input(shape=(self.flames, self.annotations, self.num_classes), dtype='int32')

        model_input = Input(shape=(self.flames,self.annotations, self.coordinates + self.num_classes))

        validity = model(model_input)

        return Model([pose_movie, label], validity)

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

    def make_categorical(y_bt,cat_dim):
        one_hot = np.zeros((y_bt.shape[0], cat_dim))
        one_hot[np.arange(y_bt.shape[0]), y_bt] = 1
        return one_hot

    def train(self, epochs, batch_size=32, save_interval=50):
        #from pose_utils import load_pose_cords_from_string
        input_folder = './annotations/'

        classes = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        pose_list = []
        label_list = []

        for label in classes:
            label_path = os.path.join(input_folder, label)
            data_num = len(os.listdir(label_path))
            label_idx = class_to_idx[label]
            label_np = np.full((data_num, 1), label_idx, dtype=np.int32)
            label_list.append(label_np)

            annotation_list = os.listdir(label_path)

            pose_np = np.zeros((len(annotation_list), ) + self.pose_movie_shape)
            for i in range(len(annotation_list)):
                df = pd.read_csv(label_path + '/%s'% annotation_list[i], sep=':')
                df = df.sort_values('name')

                t = 0

                for index, row in df.iterrows():
                    pose_np[i][t] = self.load_pose_cords(row['keypoints_y'], row['keypoints_x'])
                    t += 1

            pose_list.append(pose_np)

        pose_train = np.concatenate(pose_list, axis=0)
        label_train = np.concatenate(label_list, axis=0)

        pose_train = pose_train / 127.5 - 1

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        f = open('gan_loss.csv','a')
        writer = csv.writer(f)
        writer.writerow(['epoch','D_loss','accuracy','G_loss'])

        for epoch in range(epochs):

            idx = np.random.randint(0, pose_train.shape[0], batch_size)
            pose_movie_train = np.zeros((batch_size, 32, 18, 2), dtype=float)

            for i in range(batch_size):
                pose_movie_train[i] = pose_train[idx[i]]
                for t in range(32):
                    pose_movie_train[i][t] = self.replace_index(pose_movie_train[i][t], True)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            pose_labels = label_train[idx]

            pose_movie_gen = self.generator.predict([noise, pose_labels])

            ol = np.asarray(func.make_categorical(pose_labels, len(classes)), dtype=np.float32)
            ol = ol.reshape(pose_labels.shape[0], len(classes), 1, 1)
            k = xp.ones((pose_labels.shape[0], len(classes), self.flames, self.annotations), dtype=np.float32)
            k = k * ol

            pose_movie_train = np.concatenate([k, pose_movie_train], axis=0)
            pose_movie_gen = np.concatenate([k, pose_movie_gen], axis=0)
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(pose_movie_train, valid)#valid=real
            d_loss_fake = self.discriminator.train_on_batch(pose_movie_gen, fake)#fake
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            writer.writerow([epoch, d_loss[0], 100*d_loss[1], g_loss])

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

        r, c = 1, self.num_classes

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        pose_movie_gen = self.generator.predict([noise, sampled_labels])#gen_img -1 - 1

        # Rescale 0 - 1
        pose_movie_gen = 0.5 * pose_movie_gen + 0.5

        # Rescale 0 - 255
        pose_movie_gen = 255 * pose_movie_gen

        pose_movie_gen = pose_movie_gen.astype('int32')

        for i in range(c):
            for t in range(32):
                pose_movie_gen[i][t] = self.replace_index(pose_movie_gen[i][t], False)

            output_path = output_folder + "epoch_%d-%d.csv" %(epoch, int(i+1))

            result_file = open(output_path, 'w')
            processed_names = set()
            print('name:keypoints_y:keypoints_x',file=result_file)

            for t in range(32):
                print('%s.jpg: %s: %s' % ('{0:02d}'.format(t), str(list(pose_movie_gen[i, t, :, 0])), str(list(pose_movie_gen[i, t, :, 1]))), file=result_file)

            result_file.flush()


if __name__ == '__main__':
    movie_gan = MOVIE_GAN()

movie_gan.train(epochs=301, batch_size=32, save_interval=100)
