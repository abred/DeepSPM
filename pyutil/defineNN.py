import tensorflow as tf
import tensorflow.contrib.slim as slim

# creates the network graph
# uses variance scaling initializes (aka xavier)
# a number of conv layers (params: cnnSetSizes)
# (sets of varying feature size, always 3x3, one set = 3 layers)
# followed by a number of fully connected layers (params: fcSizes)
# NO activation function on final output!
def defineNN(nn, images, numOut=1, isTargetNN=False, isDQN=False):
    with tf.variable_scope('inf') as scope:
        net = tf.reshape(images, [-1,
                                  nn.state_dim,
                                  nn.state_dim,
                                  1], name='deflatteni1')

        with slim.arg_scope(
            [slim.fully_connected, slim.conv2d],
            activation_fn=tf.nn.relu,
            weights_initializer=
                tf.contrib.layers.variance_scaling_initializer(
                    factor=1.0, mode='FAN_AVG', uniform=True),
            weights_regularizer=slim.l2_regularizer(nn.weightDecay),
            biases_initializer=
                tf.contrib.layers.variance_scaling_initializer(
                    factor=1.0, mode='FAN_AVG', uniform=True)):

            with slim.arg_scope(
                    [slim.conv2d], stride=1, padding='SAME',
                    normalizer_fn=nn.batchnorm,
                    normalizer_params={
                        'fused': True,
                        'is_training': nn.isTraining,
                        'updates_collections': None,
                        'decay': nn.params['batchnorm-decay'],
                        'scale': True}):
                setSizes=None
                if isDQN:
                    setSizes = nn.params['cnnSetSizesDQN'].split(",")
                else:
                    setSizes = nn.params['cnnSetSizes'].split(",")
                if setSizes != [""]:
                    for i in range(len(setSizes)):
                        net = slim.repeat(net, 3, slim.conv2d,
                                          int(setSizes[i]),
                                          [3, 3], scope='conv' + str(i+1))
                        print(net)
                        if ((nn.params['numPool'] is None and
                             i < len(setSizes)-1) or
                            (nn.params['numPool'] is not None and
                             i < nn.params['numPool'])):
                            net = slim.max_pool2d(net, [2, 2],
                                                  scope='pool' + str(i+1))
                            print(net)

            net = slim.flatten(net)

            with slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    normalizer_fn=nn.batchnorm,
                    normalizer_params={
                        'fused': True,
                        'is_training': nn.isTraining,
                        'updates_collections': None,
                        'decay': nn.params['batchnorm-decay'],
                        'scale': True}):
                fcSizes=None
                if isDQN:
                    fcSizes = nn.params['fcSizesDQN'].split(",")
                else:
                    fcSizes = nn.params['fcSizes'].split(",")

                if fcSizes != [""]:
                    for i in range(len(fcSizes)):
                        net = slim.fully_connected(net, int(fcSizes[i]),
                                                   scope='fc' + str(i))
                        print(net)
                        if nn.dropout:
                            net = slim.dropout(net,
                                               keep_prob=nn.keep_prob,
                                               is_training=nn.isTraining)
                            print(net)

            if isDQN and nn.params['duelingDQN']:
                netVal = slim.fully_connected(net, 1, activation_fn=None,
                                              scope='outVal')
                netAdv = slim.fully_connected(net, numOut, activation_fn=None,
                                              scope='outAdv')
                if nn.params['veryveryverbose']:
                    netVal = tf.Print(netVal, [netVal], "netVal:",  first_n=25)
                    netAdv = tf.Print(netAdv, [netAdv], "netAdv:",  first_n=25)
                net = netVal + netAdv - tf.reduce_mean(netAdv, axis=1,
                                                       keep_dims=True)
            else:
                net = slim.fully_connected(net, numOut, activation_fn=None,
                                        scope='out')

            if nn.params['veryveryverbose']:
                net = tf.Print(net, [net], "output:",  first_n=25)

            print(net)

            if not isTargetNN:
                nn.weight_summaries += [tf.summary.histogram('output', net)]
    return net
