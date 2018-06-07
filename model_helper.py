import os
import tensorflow as tf

import las
import utils

__all__ = [
    'las_model_fn',
]


def compute_loss(logits, targets, final_sequence_length, target_sequence_length, mode):

    assert mode != tf.estimator.ModeKeys.PREDICT

    if mode == tf.estimator.ModeKeys.TRAIN:
        target_weights = tf.sequence_mask(
            target_sequence_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)
    else:
        '''
        # Reference: https://github.com/tensorflow/nmt/issues/2
        # Note that this method always trim the tensor with larger length to shorter one, 
        # and I think it is unfair. 
        # Consider targets = [[3, 3, 2]], and logits with shape [1, 2, VOCAB_SIZE]. 
        # This method will trim targets to [[3, 3]] and compute sequence_loss on new targets and logits.
        # However, I think the truth is that the model predicts less word than ground truth does,
        # and hence, both targets and logits should be padded to the same sequence length (dimension 1)
        # to compute loss.

        current_sequence_length = tf.to_int32(
            tf.minimum(tf.shape(targets)[1], tf.shape(logits)[1]))
        targets = tf.slice(targets, begin=[0, 0],
                           size=[-1, current_sequence_length])
        logits = tf.slice(logits, begin=[0, 0, 0],
                          size=[-1, current_sequence_length, -1])
        target_weights = tf.sequence_mask(
            target_sequence_length, maxlen=current_sequence_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)
        '''

        max_ts = tf.reduce_max(target_sequence_length)
        max_fs = tf.reduce_max(final_sequence_length)

        max_sequence_length = tf.to_int32(
            tf.maximum(max_ts, max_fs))

        logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, max_fs, -1])

        # pad EOS to make targets and logits have same shape
        targets = tf.pad(targets, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(targets)[1])]], constant_values=utils.EOS_ID)
        logits = tf.pad(logits, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(logits)[1])], [0, 0]], constant_values=0)

        # find larger length between predictions and targets
        sequence_length = tf.reduce_max(
            [target_sequence_length, final_sequence_length], 0)

        target_weights = tf.sequence_mask(
            sequence_length, maxlen=max_sequence_length, dtype=tf.float32)

        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)

    return loss


def las_model_fn(features,
                 labels,
                 mode,
                 config,
                 params):

    encoder_inputs = features['encoder_inputs']
    source_sequence_length = features['source_sequence_length']

    decoder_inputs = None
    targets = None
    target_sequence_length = None

    if mode != tf.estimator.ModeKeys.PREDICT:
        decoder_inputs = labels['targets_inputs']
        targets = labels['targets_outputs']
        target_sequence_length = labels['target_sequence_length']

    tf.logging.info('Building listener')

    with tf.variable_scope('listener'):
        (encoder_outputs, source_sequence_length), encoder_state = las.model.listener(
            encoder_inputs, source_sequence_length, mode, params.encoder)

    tf.logging.info('Building speller')

    with tf.variable_scope('speller'):
        decoder_outputs, final_context_state, final_sequence_length = las.model.speller(
            encoder_outputs, encoder_state, decoder_inputs,
            source_sequence_length, target_sequence_length,
            mode, params.decoder)

    with tf.name_scope('prediction'):
        if mode == tf.estimator.ModeKeys.PREDICT and params.decoder.beam_width > 0:
            logits = tf.no_op()
            sample_ids = decoder_outputs.predicted_ids
        else:
            logits = decoder_outputs.rnn_output
            sample_ids = tf.to_int32(tf.argmax(logits, -1))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'sample_ids': sample_ids,
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope('metrics'):
        edit_distance = utils.edit_distance(
            sample_ids, targets, utils.EOS_ID, params.mapping)

        metrics = {
            'edit_distance': tf.metrics.mean(edit_distance),
        }

    tf.summary.scalar('edit_distance', metrics['edit_distance'][1])

    with tf.name_scope('cross_entropy'):
        loss = compute_loss(
            logits, targets, final_sequence_length, target_sequence_length, mode)

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('alignment'):
            attention_images = utils.create_attention_images(
                final_context_state)

        attention_summary = tf.summary.image(
            'attention_images', attention_images)

        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=10,
            output_dir=os.path.join(config.model_dir, 'eval'),
            summary_op=attention_summary)

        logging_hook = tf.train.LoggingTensorHook({
            'edit_distance': tf.reduce_mean(edit_distance),
            'max_edit_distance': tf.reduce_max(edit_distance),
            'max_predictions': sample_ids[tf.argmax(edit_distance)],
            'max_targets': targets[tf.argmax(edit_distance)],
            'min_edit_distance': tf.reduce_min(edit_distance),
            'min_predictions': sample_ids[tf.argmin(edit_distance)],
            'min_targets': targets[tf.argmin(edit_distance)],
        }, every_n_iter=10)

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[logging_hook, eval_summary_hook])

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook({
        'loss': loss,
        'edit_distance': tf.reduce_mean(edit_distance),
        #'max_edit_distance': tf.reduce_max(edit_distance),
        #'predictions': sample_ids[tf.argmax(edit_distance)],
        #'targets': targets[tf.argmax(edit_distance)],
    }, every_n_secs=10)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
