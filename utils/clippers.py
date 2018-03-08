import tensorflow as tf

from dp.clippers.grouped import GroupedClipper
from dp.clippers.basic import BasicClipper
from dp.clippers.bounds import TensorBound, ConstantBound
from dp.clippers.autogroup import AutoGroupClipper
from dp.samplers.sampler import Sampler


def get_clipper(name, config):
    if name == "basic":
        return BasicClipper(ConstantBound(config.C))

    elif name == "mnist":
        def callback_b1(clipper, sess, total_step):
            # if total_step > 150:
            #     val = 3.0
            # elif total_step > 75:
            #     val = 6.0 - (6.0 - 3.0) / 75 * (total_step - 75)
            # else:
            #     val = 12.0 - (12.0 - 6.0) / 75 * total_step

            val = config.C

            return {clipper.tensor: val}

        def callback_b2(clipper, sess, total_step):
            # if total_step > 150:
            #     val = 0.75
            # elif total_step > 75:
            #     val = 1.5 - (1.5 - 0.75) / 75 * (total_step - 75)
            # else:
            #     val = 3.0 - (3.0 - 1.5) / 75 * total_step

            val = config.C

            return {clipper.tensor: val}

        b1 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b1)
        b2 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b2)

        return GroupedClipper([
            (['discriminator/Conv2D/W:0'], b1),
            (['discriminator/Conv2D_1/W:0'], b1),
            (['discriminator/Conv2D_2/W:0'], b1),
            (['discriminator/FullyConnected/W:0'], b1),
            (['discriminator/Conv2D/b:0', 'discriminator/Conv2D_1/b:0',
              'discriminator/Conv2D_2/b:0'], b2)
        ])

    elif name == "lsun":
        def callback_b1(clipper, sess, total_step):
            return {clipper.tensor: config.C}

        def callback_b2(clipper, sess, total_step):
            return {clipper.tensor: config.C}

        b1 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b1)
        b2 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b2)

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b1),
            (['discriminator/conv3/W:0'], b1),
            (['discriminator/conv4/W:0'], b1),
            (['discriminator/FullyConnected/W:0'], b1),
            (['discriminator/conv1/b:0', 'discriminator/conv2/b:0',
              'discriminator/conv3/b:0', 'discriminator/conv4/b:0'], b2)
        ])

    elif name == "lsun_simple":

        def callback_b1(clipper, sess, total_step):
            if total_step > 5000:
                val = 4.0
            elif total_step > 500:
                val = 8.0 - (8.0 - 4.0) / 4500 * (total_step - 4500)
            else:
                val = 16.0 - (16.0 - 8.0) / 500 * total_step

            return {clipper.tensor: val}

        def callback_b2(clipper, sess, total_step):
            if total_step > 5000:
                val = 5.6
            elif total_step > 500:
                val = 11.2 - (11.2 - 5.6) / 4500 * (total_step - 4500)
            else:
                val = 22.4 - (22.4 - 11.2) / 500 * total_step

            return {clipper.tensor: val}

        def callback_b3(clipper, sess, total_step):
            if total_step > 5000:
                val = 1.0
            elif total_step > 500:
                val = 2.0 - (2.0 - 1.0) / 4500 * (total_step - 4500)
            else:
                val = 4.0 - (4.0 - 2.0) / 500 * total_step

            return {clipper.tensor: val}

        b1 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b1)
        b2 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b2)
        b3 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b3)

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b1),
            (['discriminator/conv3/W:0'], b2),
            (['discriminator/conv4/W:0'], b2),
            (['discriminator/FullyConnected/W:0'], b2),
            (['discriminator/conv1/b:0', 'discriminator/conv2/b:0',
              'discriminator/conv3/b:0', 'discriminator/conv4/b:0'], b3)
        ])

    elif name == "lsun_small":
        def callback_b1(clipper, sess, total_step):
            if total_step > 5000:
                val = 4.0
            elif total_step > 500:
                val = 8.0 - (8.0 - 4.0) / 4500 * (total_step - 4500)
            else:
                val = 16.0 - (16.0 - 8.0) / 500 * total_step

            return {clipper.tensor: val}

        def callback_b2(clipper, sess, total_step):
            if total_step > 5000:
                val = 5.6
            elif total_step > 500:
                val = 11.2 - (11.2 - 5.6) / 4500 * (total_step - 4500)
            else:
                val = 22.4 - (22.4 - 11.2) / 500 * total_step

            return {clipper.tensor: val}

        def callback_b3(clipper, sess, total_step):
            if total_step > 5000:
                val = 1.0
            elif total_step > 500:
                val = 2.0 - (2.0 - 1.0) / 4500 * (total_step - 4500)
            else:
                val = 4.0 - (4.0 - 2.0) / 500 * total_step

            return {clipper.tensor: val}

        b1 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b1)
        b2 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b2)
        b3 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b3)

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b1),
            (['discriminator/conv3/W:0'], b1),
            (['discriminator/conv4/W:0'], b2),
            (['discriminator/FullyConnected/W:0'], b2),
            (['discriminator/conv1/b:0', 'discriminator/conv2/b:0',
              'discriminator/conv3/b:0', 'discriminator/conv4/b:0'], b3)
        ])

    elif name == "lsun_est_simple":
        sampler = Sampler(1)
        b1 = sampler.get_bound_tensor("discriminator/conv1/W:0")
        b2 = sampler.get_bound_tensor("discriminator/conv2/W:0")
        b3 = sampler.get_bound_tensor("discriminator/conv3/W:0")
        b4 = sampler.get_bound_tensor("discriminator/conv4/W:0")
        b5 = sampler.get_bound_tensor("discriminator/FullyConnected/W:0")
        b6 = sampler.get_bound_tensor("discriminator/conv1/b:0")
        b7 = sampler.get_bound_tensor("discriminator/conv2/b:0")
        b8 = sampler.get_bound_tensor("discriminator/conv3/b:0")
        b9 = sampler.get_bound_tensor("discriminator/conv4/b:0")

        b1 = TensorBound(b1)
        b2 = TensorBound(b2)
        b3 = TensorBound(b3)
        b4 = TensorBound(b4)
        b5 = TensorBound(b5)
        b6 = TensorBound(b6)
        b7 = TensorBound(b7)
        b8 = TensorBound(b8)
        b9 = TensorBound(b9)

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b2),
            (['discriminator/conv3/W:0'], b3),
            (['discriminator/conv4/W:0'], b4),
            (['discriminator/FullyConnected/W:0'], b5),
            (["discriminator/conv1/b:0"], b6),
            (['discriminator/conv2/b:0'], b7),
            (['discriminator/conv3/b:0'], b8),
            (['discriminator/conv4/b:0'], b9)
        ]), sampler

    elif name == "lsun_est":
        sampler = Sampler(1)
        b1 = sampler.get_bound_tensor("discriminator/conv1/W:0")
        b2 = sampler.get_bound_tensor("discriminator/conv2/W:0")
        b3 = sampler.get_bound_tensor("discriminator/conv3/W:0")
        b4 = sampler.get_bound_tensor("discriminator/conv4/W:0")
        b5 = sampler.get_bound_tensor("discriminator/FullyConnected/W:0")
        b6 = sampler.get_bound_tensor("discriminator/conv1/b:0")
        b7 = sampler.get_bound_tensor("discriminator/conv2/b:0")
        b8 = sampler.get_bound_tensor("discriminator/conv3/b:0")
        b9 = sampler.get_bound_tensor("discriminator/conv4/b:0")

        b1 = TensorBound(b1)
        b2 = TensorBound(b2)
        b3 = TensorBound(b3)
        b4 = TensorBound(b4)
        b5 = TensorBound(b5)
        b6 = TensorBound(tf.sqrt(tf.square(b6) + tf.square(b7)))
        b7 = TensorBound(tf.sqrt(tf.square(b8) + tf.square(b9)))

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b2),
            (['discriminator/conv3/W:0'], b3),
            (['discriminator/conv4/W:0'], b4),
            (['discriminator/FullyConnected/W:0'], b5),
            (['discriminator/conv1/b:0', 'discriminator/conv2/b:0'], b6),
            (['discriminator/conv3/b:0', 'discriminator/conv4/b:0'], b7)
        ]), sampler

    elif name == "mnist_est_simple":
        sampler = Sampler(1)
        b1 = sampler.get_bound_tensor("discriminator/Conv2D/W:0")
        b2 = sampler.get_bound_tensor("discriminator/Conv2D_1/W:0")
        b3 = sampler.get_bound_tensor("discriminator/Conv2D_2/W:0")
        b4 = sampler.get_bound_tensor("discriminator/FullyConnected/W:0")
        b5 = sampler.get_bound_tensor("discriminator/Conv2D/b:0")
        b6 = sampler.get_bound_tensor("discriminator/Conv2D_1/b:0")
        b7 = sampler.get_bound_tensor("discriminator/Conv2D_2/b:0")

        b1 = TensorBound(b1)
        b2 = TensorBound(b2)
        b3 = TensorBound(b3)
        b4 = TensorBound(b4)
        b5 = TensorBound(b5)
        b6 = TensorBound(b6)
        b7 = TensorBound(b7)

        return GroupedClipper([
            (['discriminator/Conv2D/W:0'], b1),
            (['discriminator/Conv2D_1/W:0'], b2),
            (['discriminator/Conv2D_2/W:0'], b3),
            (['discriminator/FullyConnected/W:0'], b4),
            (['discriminator/Conv2D/b:0'], b5),
            (["discriminator/Conv2D_1/b:0"], b6),
            (["discriminator/Conv2D_2/b:0"], b7),
        ]), sampler

    elif name == "mnist_est":
        sampler = Sampler(1)
        b1 = sampler.get_bound_tensor("discriminator/Conv2D/W:0")
        b2 = sampler.get_bound_tensor("discriminator/Conv2D_1/W:0")
        b3 = sampler.get_bound_tensor("discriminator/Conv2D_2/W:0")
        b4 = sampler.get_bound_tensor("discriminator/FullyConnected/W:0")
        b5 = sampler.get_bound_tensor("discriminator/Conv2D/b:0")
        b6 = sampler.get_bound_tensor("discriminator/Conv2D_1/b:0")
        b7 = sampler.get_bound_tensor("discriminator/Conv2D_2/b:0")

        b1 = TensorBound(b1)
        b2 = TensorBound(b2)
        b3 = TensorBound(b3)
        b4 = TensorBound(b4)
        b5 = TensorBound(tf.sqrt(tf.square(b5) + tf.square(b6) + tf.square(b7)))

        return GroupedClipper([
            (['discriminator/Conv2D/W:0'], b1),
            (['discriminator/Conv2D_1/W:0'], b2),
            (['discriminator/Conv2D_2/W:0'], b3),
            (['discriminator/FullyConnected/W:0'], b4),
            (['discriminator/Conv2D/b:0', 'discriminator/Conv2D_1/b:0',
              'discriminator/Conv2D_2/b:0'], b5),
        ]), sampler

    elif name == "cifar10_est":
        sampler = Sampler(1)
        b1 = sampler.get_bound_tensor("discriminator/conv1/W:0")
        b2 = sampler.get_bound_tensor("discriminator/conv2/W:0")
        b3 = sampler.get_bound_tensor("discriminator/conv3/W:0")
        b4 = sampler.get_bound_tensor("discriminator/conv4/W:0")
        b5 = sampler.get_bound_tensor("discriminator/FullyConnected/W:0")
        b6 = sampler.get_bound_tensor("discriminator/conv1/b:0")
        b7 = sampler.get_bound_tensor("discriminator/conv2/b:0")
        b8 = sampler.get_bound_tensor("discriminator/conv3/b:0")
        b9 = sampler.get_bound_tensor("discriminator/conv4/b:0")

        b1 = TensorBound(tf.sqrt(tf.square(b1) + tf.square(b2)))
        b2 = TensorBound(tf.sqrt(tf.square(b3) + tf.square(b4)))
        b3 = TensorBound(b5)
        b4 = TensorBound(tf.sqrt(tf.square(b6) + tf.square(b7) + tf.square(b8)
                                 + tf.square(b9)))

        return GroupedClipper([
            (['discriminator/conv1/W:0', 'discriminator/conv2/W:0'], b1),
            (['discriminator/conv3/W:0', 'discriminator/conv4/W:0'], b2),
            (['discriminator/FullyConnected/W:0'], b3),
            (['discriminator/conv1/b:0', 'discriminator/conv2/b:0',
              'discriminator/conv3/b:0', 'discriminator/conv4/b:0'], b4),
        ]), sampler

    elif name == "mnist_ag_6":
        sampler = Sampler(1, keep_memory=False)

        def callback(sess, steps):
            return {k: b for k, b in sampler._bounds.items()}

        return AutoGroupClipper(6, callback), sampler

    elif name == "mnist_ag_5":
        sampler = Sampler(1, keep_memory=False)

        def callback(sess, steps):
            return {k: b for k, b in sampler._bounds.items()}

        return AutoGroupClipper(5, callback), sampler

    elif name == "mnist_ag_4":
        sampler = Sampler(1, keep_memory=False)

        def callback(sess, steps):
            return {k: b for k, b in sampler._bounds.items()}

        return AutoGroupClipper(4, callback), sampler

    elif name == "mnist_ag_3":
        sampler = Sampler(1, keep_memory=False)

        def callback(sess, steps):
            return {k: b for k, b in sampler._bounds.items()}

        return AutoGroupClipper(3, callback), sampler

    elif name == "lsun_ag_7":
        sampler = Sampler(1, keep_memory=False)

        def callback(sess, steps):
            return {k: b for k, b in sampler._bounds.items()}

        return AutoGroupClipper(7, callback), sampler

    elif name == "celeba_est":
        sampler = Sampler(1)
        b1 = sampler.get_bound_tensor("discriminator/conv1/W:0")
        b2 = sampler.get_bound_tensor("discriminator/conv2/W:0")
        b3 = sampler.get_bound_tensor("discriminator/conv3/W:0")
        b4 = sampler.get_bound_tensor("discriminator/conv4/W:0")
        b5 = sampler.get_bound_tensor("discriminator/FullyConnected/W:0")
        b6 = sampler.get_bound_tensor("discriminator/conv1/b:0")
        b7 = sampler.get_bound_tensor("discriminator/conv2/b:0")
        b8 = sampler.get_bound_tensor("discriminator/conv3/b:0")
        b9 = sampler.get_bound_tensor("discriminator/conv4/b:0")

        b1 = TensorBound(b1)
        b2 = TensorBound(b2)
        b3 = TensorBound(b3)
        b4 = TensorBound(b4)
        b5 = TensorBound(b5)
        b6 = TensorBound(tf.sqrt(tf.square(b6) + tf.square(b7)))
        b7 = TensorBound(tf.sqrt(tf.square(b8) + tf.square(b9)))

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b2),
            (['discriminator/conv3/W:0'], b3),
            (['discriminator/conv4/W:0'], b4),
            (['discriminator/FullyConnected/W:0'], b5),
            (['discriminator/conv1/b:0', 'discriminator/conv2/b:0'], b6),
            (['discriminator/conv3/b:0', 'discriminator/conv4/b:0'], b7)
        ]), sampler

    elif name == "celeba_48":
        def callback_b1(clipper, sess, total_step):
            return {clipper.tensor: config.C}

        def callback_b2(clipper, sess, total_step):
            return {clipper.tensor: config.C}

        b1 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b1)
        b2 = TensorBound(tf.placeholder(tf.float32, shape=()), update_callback=callback_b2)

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b1),
            (['discriminator/conv3/W:0'], b1),
            (['discriminator/FullyConnected/W:0'], b1),
            (['discriminator/conv1/b:0', 'discriminator/conv2/b:0',
              'discriminator/conv3/b:0'], b2)
        ])

    elif name == "celeba_48_est_simple":
        sampler = Sampler(1)
        b1 = sampler.get_bound_tensor("discriminator/conv1/W:0")
        b2 = sampler.get_bound_tensor("discriminator/conv2/W:0")
        b3 = sampler.get_bound_tensor("discriminator/conv3/W:0")
        b5 = sampler.get_bound_tensor("discriminator/FullyConnected/W:0")
        b6 = sampler.get_bound_tensor("discriminator/conv1/b:0")
        b7 = sampler.get_bound_tensor("discriminator/conv2/b:0")
        b8 = sampler.get_bound_tensor("discriminator/conv3/b:0")

        b1 = TensorBound(b1)
        b2 = TensorBound(b2)
        b3 = TensorBound(b3)
        b5 = TensorBound(b5)
        b6 = TensorBound(b6)
        b7 = TensorBound(b7)
        b8 = TensorBound(b8)

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b2),
            (['discriminator/conv3/W:0'], b3),
            (['discriminator/FullyConnected/W:0'], b5),
            (['discriminator/conv1/b:0'], b6),
            (['discriminator/conv2/b:0'], b7),
            (["discriminator/conv3/b:0"], b8),
        ]), sampler

    elif name == "celeba_48_est":
        sampler = Sampler(1)
        b1 = sampler.get_bound_tensor("discriminator/conv1/W:0")
        b2 = sampler.get_bound_tensor("discriminator/conv2/W:0")
        b3 = sampler.get_bound_tensor("discriminator/conv3/W:0")
        b5 = sampler.get_bound_tensor("discriminator/FullyConnected/W:0")
        b6 = sampler.get_bound_tensor("discriminator/conv1/b:0")
        b7 = sampler.get_bound_tensor("discriminator/conv2/b:0")
        b8 = sampler.get_bound_tensor("discriminator/conv3/b:0")

        b1 = TensorBound(b1)
        b2 = TensorBound(b2)
        b3 = TensorBound(b3)
        b5 = TensorBound(b5)
        b6 = TensorBound(b6)
        b7 = TensorBound(tf.sqrt(tf.square(b7) + tf.square(b8)))

        return GroupedClipper([
            (['discriminator/conv1/W:0'], b1),
            (['discriminator/conv2/W:0'], b2),
            (['discriminator/conv3/W:0'], b3),
            (['discriminator/FullyConnected/W:0'], b5),
            (['discriminator/conv1/b:0'], b6),
            (['discriminator/conv2/b:0', 'discriminator/conv3/b:0'], b7)
        ]), sampler

    elif name == "celeba_ag_5":
        sampler = Sampler(1, keep_memory=False)

        def callback(sess, steps):
            return {k: b for k, b in sampler._bounds.items()}

        return AutoGroupClipper(5, callback), sampler

    elif name == "celeba_48_ag_6":
        sampler = Sampler(1, keep_memory=False)

        def callback(sess, steps):
            return {k: b for k, b in sampler._bounds.items()}

        return AutoGroupClipper(6, callback), sampler

    elif name == "no":
        from dp.clippers.no import NoClipper
        return NoClipper()

    raise NotImplementedError()