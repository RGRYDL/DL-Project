from __future__ import division
import os
import time 
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *

np.set_printoptions(threshold=np.inf)

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        self.generator = generator

        self.abs_criterion = abs_criterion #一范数
        self.mae_criterion = mae_criterion #二范数
        self.sce_criterion = sce_criterion #sigmoid交叉熵
        self.cross_entropy = cross_entropy #交叉熵
        
        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + 1 + self.output_c_dim],
                                        name='real_A_B_C_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + 1]
        self.real_C = self.real_data[:, :, :, self.input_c_dim + 1:self.input_c_dim + 1 + self.output_c_dim]

        self.real_B = tf.add(self.real_B, tf.ones_like(self.real_B)) / tf.constant(2.)

        #g_loss
        self.fake_B = self.generator(self.real_A, self.options, 1, False, name="generatorA2B")
        self.fake_B = tf.add(self.fake_B, tf.ones_like(self.fake_B)) / tf.constant(2.)

        self.fake_C = self.generator(tf.concat([self.real_A,self.fake_B], 3), self.options, self.options.output_c_dim, False, name="generatorAB2C")

        self.DB_fake = self.discriminator(tf.concat([self.real_A,self.fake_B], 3), self.options, reuse=False, name="discriminatorB")
        self.DC_fake = self.discriminator(tf.concat([self.real_A,self.fake_B,self.fake_C], 3), self.options, reuse=False, name="discriminatorC")

        self.g_loss = sce_criterion(self.fake_B, self.real_B) \
            + self.lambda1 * abs_criterion(self.fake_C, self.real_C) \
            + self.lambda2 * sce_criterion(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.lambda3 * sce_criterion(self.DC_fake, tf.ones_like(self.DC_fake))
        self.g_loss1 = sce_criterion(self.fake_B, self.real_B)

        #d_loss
        self.fake_B_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='fake_B_sample')
        self.fake_C_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.options.output_c_dim], name='fake_C_sample')

        self.DB_real = self.discriminator(tf.concat([self.real_A,self.real_B], 3), self.options, reuse=True, name="discriminatorB")
        self.DC_real = self.discriminator(tf.concat([self.real_A,self.real_B,self.real_C], 3), self.options, reuse=True, name="discriminatorC")
        self.DB_fake_sample = self.discriminator(tf.concat([self.real_A,self.fake_B_sample], 3), self.options, reuse=True, name="discriminatorB")
        self.DC_fake_sample = self.discriminator(tf.concat([self.real_A,self.fake_B_sample,self.fake_C_sample], 3), self.options, reuse=True, name="discriminatorC")

        self.db_loss_real = self.sce_criterion(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.sce_criterion(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = self.db_loss_real + self.db_loss_fake
        self.dc_loss_real = self.sce_criterion(self.DC_real, tf.ones_like(self.DC_real))
        self.dc_loss_fake = self.sce_criterion(self.DC_fake_sample, tf.zeros_like(self.DC_fake_sample))
        self.dc_loss = self.dc_loss_real + self.dc_loss_fake
        self.d_loss = self.lambda2 * self.db_loss + self.lambda3 * self.dc_loss

        #summary
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_sum])
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_sum = tf.summary.merge([self.d_loss_sum]
        )

        #test
        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')

        self.testB = self.generator(self.test_A, self.options, 1, True, name="generatorA2B")
        self.testB = tf.add(self.testB, tf.ones_like(self.testB)) / tf.constant(2.)

        self.testC = self.generator(tf.concat([self.test_A,self.testB], 3), self.options, self.options.output_c_dim, True, name="generatorAB2C")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train ST_GAN"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/train/train_A'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/train/train_B'))
            dataC = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/train/train_C'))

            #np.random.shuffle(dataA)
            #np.random.shuffle(dataB)
            #np.random.shuffle(dataC)

            batch_idxs = min(min(len(dataA), len(dataB), len(dataC)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataC[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                fake_B, fake_C = self.sess.run(
                    [self.fake_B, self.fake_C],
                    feed_dict={self.real_data: batch_images})

				# Update D network
                d_loss, _, summary_str = self.sess.run(
                    [self.d_loss, self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_B_sample: fake_B,
                               self.fake_C_sample: fake_C,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                # Update G network and record fake outputs
                g_loss1, g_loss, _, summary_str = self.sess.run(
                    [self.g_loss1, self.g_loss, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                [fake_B, fake_C] = self.pool([fake_B, fake_C])

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %4.4f g_loss: %4.4f g_loss1: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss, g_loss1)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                #if np.mod(counter, args.save_freq) == 2:
                    #self.save(args.checkpoint_dir, counter)
                if (np.mod(epoch + 1, 25) == 0) and (idx == batch_idxs - 1):
                	self.save(args.checkpoint_dir, epoch + 1)

    def save(self, checkpoint_dir, step):
        model_name = "ST_GAN.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/train/train_A'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/train/train_B'))
        dataC = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/train/train_C'))
        np.random.shuffle(dataA)
        #np.random.shuffle(dataB)
        #np.random.shuffle(dataC)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size], dataC[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        real_A, fake_B, fake_C = self.sess.run(
            [self.real_A, self.fake_B, self.fake_C],
            feed_dict={self.real_data: sample_images}
        )

        fake_B = 2. * fake_B - 1.

        save_images(real_A, [self.batch_size, 1],
                    './{}/{:02d}_{:04d}_A.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/{:02d}_{:04d}_B.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_C, [self.batch_size, 1],
                    './{}/{:02d}_{:04d}_C.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test ST_GAN"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/test/test_A'))

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, 'test_index.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>outputB</th><th>outputC</th></tr>")

        out_varB, out_varC, in_var = (self.testB, self.testC, self.test_A)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)

            image_path = os.path.join(args.test_dir, 'test_{0}'.format(os.path.basename(sample_file)))
            image_pathB = os.path.join(args.test_dir, 'test_{0}_B.png'.format(os.path.splitext(os.path.basename(sample_file))[0]))
            image_pathC = os.path.join(args.test_dir, 'test_{0}_C.png'.format(os.path.splitext(os.path.basename(sample_file))[0]))

            fake_imgB, fake_imgC = self.sess.run([out_varB, out_varC], feed_dict={in_var: sample_image})
            
            fake_imgB = 2. * fake_imgB - 1.

            save_images(fake_imgB, [1, 1], image_pathB)
            save_images(fake_imgC, [1, 1], image_pathC)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_pathB if os.path.isabs(image_pathB) else (
                '..' + os.path.sep + image_pathB)))
            index.write("<td><img src='%s'></td>" % (image_pathC if os.path.isabs(image_pathC) else (
                '..' + os.path.sep + image_pathC)))
            index.write("</tr>")
        index.close()
