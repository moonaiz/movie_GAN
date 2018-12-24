from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape
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

    def load_pose_cords_from_strings(y_str, x_str):
        y_cords = json.loads(y_str)
        x_cords = json.loads(x_str)
        return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

    def train(self, epochs, batch_size=32, save_interval=50):
        input_folder = './annotations/'
        annotation_list = os.listdir(input_folder)

        for i in range(len(annotation_list)):
            df = pd.read_csv(input_folder + '%s.csv'% annotation_list[i], sep=':')

            for index, row in df.iterrows():
                train[i] = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])

        train = train / 127.5 - 1

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            pose_movie_train = train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            pose_movie_gen = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(pose_movie_train, valid)#valid=real より全て1
            d_loss_fake = self.discriminator.train_on_batch(pose_movie_gen, fake)#fakeは全て0
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)#配列の中身の足し算

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_annotations(epoch)

    def save_annotations(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        pose_movie_gen = self.generator.predict(noise)#gen_img -1 - 1

        output_folder = './output/annotations/'
        if not os.path.exist(output_folder):
            os.mkdir(output_folder)

        # Rescale 0 - 1
        pose_movie_gen = 0.5 * pose_movie_gen + 0.5

        # Rescale 0 - 255
        pose_movie_gen = 255 * pose_movie_gen

        output_path = output_folder + "epoch_%d.csv" % epoch

        if os.path.exists(output_path):
            processed_names = set(pd.read_csv(output_path, sep=':')['name'])
            result_file = open(output_path, 'a')
        else:
            result_file = open(output_path, 'w')
            processed_names = set()
            print >> result_file, 'name:keypoints_y:keypoints_x'

        print >> result_file, "%s: %s: %s" % ('result', str(list(pose_movie_gen[:, 0])), str(list(pose_movie_gen[:, 1])))
        result_file.flush()



 if __name__ == '__main__':
     movie_gan = MOVIE_GAN()
     
 movie_gan.train(epochs=500, batch_size=32, save_interval=50)
