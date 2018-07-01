import tensorflow as tf

__all__ = [
    'lstm_cell',
    'bilstm',
    'pyramidal_bilstm',
]


def lstm_cell(num_units, dropout, mode):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    if dropout > 0.0:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=cell, input_keep_prob=(1.0 - dropout))

    return cell


def bilstm(inputs,
           sequence_length,
           num_units,
           dropout,
           mode):

    with tf.variable_scope('fw_cell'):
        forward_cell = lstm_cell(num_units, dropout, mode)
    with tf.variable_scope('bw_cell'):
        backward_cell = lstm_cell(num_units, dropout, mode)

    return tf.nn.bidirectional_dynamic_rnn(
        forward_cell,
        backward_cell,
        inputs,
        sequence_length=sequence_length,
        dtype=tf.float32)


def pyramidal_stack(outputs, sequence_length):
    shape = tf.shape(outputs)
    batch_size, max_time = shape[0], shape[1]
    num_units = outputs.get_shape().as_list()[-1]
    paddings = [[0, 0], [0, tf.floormod(max_time, 2)], [0, 0]]
    outputs = tf.pad(outputs, paddings)

    '''
    even_time = outputs[:, ::2, :]
    odd_time = outputs[:, 1::2, :]

    concat_outputs = tf.concat([even_time, odd_time], -1)
    '''

    concat_outputs = tf.reshape(outputs, (batch_size, -1, num_units * 2))

    return concat_outputs, tf.floordiv(sequence_length, 2) + tf.floormod(sequence_length, 2)


def pyramidal_bilstm(inputs,
                     sequence_length,
                     mode,
                     hparams):

    outputs = inputs

    for layer in range(hparams.num_layers):
        with tf.variable_scope('bilstm_{}'.format(layer)):
            outputs, state = bilstm(
                outputs, sequence_length, hparams.num_units, hparams.dropout, mode)

            outputs = tf.concat(outputs, -1)

            if layer != 0:
                outputs, sequence_length = pyramidal_stack(
                    outputs, sequence_length)

    return (outputs, sequence_length), state
