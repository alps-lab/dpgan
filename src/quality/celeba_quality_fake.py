#!/usr/bin/env python
from six.moves import xrange, cPickle

import tempfile
import os
import argparse

import tflearn
import numpy as np
import tensorflow as tf
from tqdm import trange

from tasks.quality_score_unlabeled import get_quality_score
from models.gans.d48_resnet_dcgan import generator_forward
from models.tasks.lsun_bedroom import classifier_forward
from utils.data_mp import CelebALoader, celeba_process_actions


def images_iter(names):
    for i, name in enumerate(names):
        images = np.load(name)
        images = list(images)

        for img in images:
            yield img


def main(config):
    config.dim = 64
    num_step = config.num_step
    fake_images = generator_forward(config, name="generator")
    input_labels = tf.placeholder(tf.float32, [None, 1], name="input_labels")
    input_images = tf.placeholder(tf.float32, [None, 48, 48, 3], name="input_images")
    logits = classifier_forward(config, input_images, name="classifier")
    probs = tf.nn.sigmoid(logits)

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=input_labels, logits=logits))
    classifier_vars = [var for var in tf.global_variables()
                                                    if var.name.startswith("classifier")]
    train_step = tf.train.AdamOptimizer().minimize(loss,
                                                   var_list=[var for var in classifier_vars
                                                    if var.name.startswith("classifier")])

    names = []
    fobjs = []

    generator_saver = tf.train.Saver([var for var in tf.global_variables()
                                      if var.name.startswith("generator")
                                      and not var.name.endswith("is_training:0")])
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # load GAN weights
    generator_saver.restore(sess, config.model_path)

    # train classifier
    data_loader = CelebALoader(config.data_dir, block_size=10,
                             num_workers=4, actions=celeba_process_actions(48), dim=48)
    try:
        tflearn.is_training(True, sess)
        data_loader.start_fetch()
        bar = trange(config.x)
        for _ in bar:
            real_images, _ = data_loader.next_batch(config.batch_size)
            real_labels = np.full([len(real_images), 1], 1, np.float32)

            generated_images = sess.run(fake_images)
            generated_labels = np.full([len(generated_images), 1], 0, np.float32)

            images = np.concatenate([real_images, generated_images], axis=0)
            labels = np.concatenate([real_labels, generated_labels], axis=0)

            indices = np.random.permutation(np.arange(0, len(images), dtype=np.int64))
            images = images[indices]
            labels = labels[indices]

            # print(images.shape, labels.shape)
            # print(input_images, input_labels)

            loss_value, _ = sess.run([loss, train_step], feed_dict={
                input_labels: labels,
                input_images: images
            })
            bar.set_description("loss: %.4f" % loss_value)

    finally:
        data_loader.stop_fetch()


    try:
        tflearn.is_training(False, sess)
        for _ in trange(num_step):
            fd, name = tempfile.mkstemp(suffix=".npy")
            fobj = os.fdopen(fd, "wb+")
            names.append(name)
            fobjs.append(fobj)
            image_arr = sess.run(fake_images)
            np.save(fobj, image_arr, allow_pickle=False)
            fobj.close()

        mean_score, std_score = get_quality_score(
            sess, input_images, probs,
            images_iter(names),
            batch_size=100,
            split=10)

        print("mean = %.4f, std = %.4f." % (mean_score, std_score))

        if config.save_path is not None:
            with open(config.save_path, "wb") as f:
                cPickle.dump(dict(batch_size=config.batch_size,
                                  scores=dict(mean=mean_score, std=std_score)), f)
    finally:
        for name in names:
            os.unlink(name)
        for fobj in fobjs:
            fobj.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", metavar="MODELPATH")
    parser.add_argument("--gen-dim", default=64, type=int, dest="gen_dim")
    parser.add_argument("-d", "--data-dir", dest="data_dir")
    parser.add_argument("--steps", dest="num_step",type=int, default=2000)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=100)
    parser.add_argument("--save-path", dest="save_path")
    parser.add_argument("-x", type=int, default=1000)
    main(parser.parse_args())
