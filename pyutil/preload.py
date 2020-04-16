import tensorflow as tf

# preload training and val data for classifier to gpu memory
class Preloader:
    def __init__(self,
                 img_train, l_train,
                 img_val, l_val, f_val):
        self.img_train = img_train
        self.l_train = l_train
        self.img_val = img_val
        self.l_val = l_val
        self.f_val = f_val

    # setup queues/input_produces etc
    def setupPreload(self, params):
        cropSz = params['pxRes']

        self.images_initializer_train = tf.placeholder(
            dtype=self.img_train.dtype,
            shape=self.img_train.shape)
        self.labels_initializer_train = tf.placeholder(
            dtype=self.l_train.dtype,
            shape=self.l_train.shape)
        self.input_images_train = tf.Variable(
            self.images_initializer_train, trainable=False, collections=[])
        self.input_labels_train = tf.Variable(
            self.labels_initializer_train, trainable=False, collections=[])

        self.images_initializer_val = []
        self.labels_initializer_val = []
        self.files_initializer_val = []
        self.input_images_val = []
        self.input_labels_val = []
        self.input_files_val = []

        for i in range(len(self.img_val)):
            self.images_initializer_val.append(tf.placeholder(
                dtype=self.img_val[i].dtype,
                shape=self.img_val[i].shape))
            self.labels_initializer_val.append(tf.placeholder(
                dtype=self.l_val[i].dtype,
                shape=self.l_val[i].shape))
            self.files_initializer_val.append(tf.placeholder(
                dtype=self.f_val[i].dtype,
                shape=self.f_val[i].shape))
            self.input_images_val.append(tf.Variable(
                self.images_initializer_val[i],
                trainable=False, collections=[]))
            self.input_labels_val.append(tf.Variable(
                self.labels_initializer_val[i],
                trainable=False, collections=[]))
            self.input_files_val.append(tf.Variable(
                self.files_initializer_val[i],
                trainable=False, collections=[]))

        # takes array of data and returns generator that produces samples
        # shuffles input
        image_train, label_train = \
            tf.train.slice_input_producer(
                [self.input_images_train,
                 self.input_labels_train],
                num_epochs=params['numTrainSteps'], shuffle=True)

        image_val = []
        label_val = []
        file_val = []
        for i in range(len(self.img_val)):
             it, lt, ft = \
                tf.train.slice_input_producer(
                    [self.input_images_val[i],
                     self.input_labels_val[i],
                     self.input_files_val[i]] , shuffle=True)
             image_val.append(it)
             label_val.append(lt)
             file_val.append(ft)

        # data augmentation 1
        # at each step, select random subimage of size cropSz
        image_train = tf.random_crop(image_train,
                                     [cropSz, cropSz, 1])

        # data augmentation 2
        # flip image and optionally distort it
        if params['distorted']:
            # Randomly flip the image horizontally.
            distorted_img_train = tf.image.random_flip_left_right(
                image_train)
            distorted_img_train = tf.image.random_flip_up_down(
                distorted_img_train)

            # randomize distortion order
            delta = params['distortBrightnessRelative']
            factor = params['distortContrast']
            if delta == 0.0 and factor == 0.0:
                pass
            elif delta == 0.0 and factor != 0.0:
                print(distorted_img_train)
                mn = tf.reduce_mean(distorted_img_train)
                print(mn)
                distorted_img_train -= mn
                distorted_img_train = tf.image.random_contrast(
                    distorted_img_train,
                    lower=1.0-factor, upper=1.0+factor)
                print(distorted_img_train)
                distorted_img_train += mn
                print(distorted_img_train)
                if params['clip']:
                    distorted_img_train = tf.clip_by_value(distorted_img_train,
                                                           -1.5, 1.5)
            elif delta != 0.0 and factor == 0.0:
                distorted_img_train = tf.image.random_brightness(
                    distorted_img_train,
                    max_delta=delta)
            else: # delta == 0.0 and factor == 0.0:
                truefn = tf.image.random_brightness(
                    tf.image.random_contrast(
                        distorted_img_train,
                        lower=1.0-factor, upper=1.0+factor),
                    max_delta=delta)
                falsefn = tf.image.random_contrast(
                    tf.image.random_brightness(
                        distorted_img_train,
                        max_delta=delta),
                    lower=1.0-factor, upper=1.0+factor)
                distorted_img_train = tf.cond(tf.random_uniform(()) < 0.5,
                                              lambda: truefn,
                                              lambda: falsefn)
            print(distorted_img_train)

            stddevGauss = params['distortGaussian']
            fracSP = params['distortSaltPepper']
            if stddevGauss != 0.0:
                distorted_img_train = distorted_img_train + tf.random_normal(
                    tf.shape(distorted_img_train),
                    mean=0.,
                    stddev=stddevGauss)
                print(distorted_img_train)
            if fracSP != 0.0:
                tmp = tf.random_uniform(tf.shape(distorted_img_train))
                minVal = tf.fill(tf.shape(distorted_img_train), -1.5)
                maxVal = tf.fill(tf.shape(distorted_img_train), 1.5)
                tmp2 = tf.less(tmp,fracSP)
                tmp2 = tf.to_float(tmp2)
                tmp = tf.where(tmp < fracSP/2.0,
                               minVal,
                               maxVal)
                tmp = tmp*tmp2
                distorted_img_train = tf.where(tmp > -0.1,
                                               distorted_img_train,
                                               tmp)
                distorted_img_train = tf.where(tmp < 0.1,
                                               distorted_img_train,
                                               tmp)
                print(distorted_img_train)
            image_train = distorted_img_train

        # create mini batches
        mbs = tf.placeholder_with_default(params['miniBatchSize'],
                                          shape=(),
                                          name="miniBatchSize_pl")
        images_train, labels_train = \
            tf.train.batch(
                [image_train,
                 label_train],
                batch_size=mbs,
                name="training")

        images_val = []
        labels_val = []
        files_val = []
        for i in range(len(self.img_val)):
            image_val[i] = tf.random_crop(image_val[i],
                                           [cropSz, cropSz, 1])

            it, lt, ft = \
                tf.train.batch(
                    [image_val[i],
                     label_val[i],
                     file_val[i]],
                    batch_size=mbs,
                    allow_smaller_final_batch=True,
                    name="evaluation")
            images_val.append(it)
            labels_val.append(lt)
            files_val.append(ft)

        return images_train, labels_train, \
            images_val, labels_val, files_val

    # actually upload data
    def initPreload(self, sess):
        sess.run(self.input_images_train.initializer,
                 feed_dict={self.images_initializer_train: self.img_train})
        sess.run(self.input_labels_train.initializer,
                 feed_dict={self.labels_initializer_train: self.l_train})
        for i in range(len(self.img_val)):
            sess.run(
                self.input_images_val[i].initializer,
                feed_dict={self.images_initializer_val[i]: self.img_val[i]})
            sess.run(
                self.input_labels_val[i].initializer,
                feed_dict={self.labels_initializer_val[i]: self.l_val[i]})
            sess.run(
                self.input_files_val[i].initializer,
                feed_dict={self.files_initializer_val[i]: self.f_val[i]})
