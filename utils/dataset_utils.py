import tensorflow as tf

__all__ = [
    'read_dataset',
    'process_dataset',
]


def read_dataset(filename, num_channels=39):
    """Read data from tfrecord file."""

    def parse_fn(example_proto):
        """Parse function for reading single sequence example."""
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[num_channels], dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string)
        }

        context, sequence = tf.parse_single_sequence_example(
            serialized=example_proto,
            sequence_features=sequence_features
        )

        return sequence['inputs'], sequence['labels']

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_fn)

    return dataset


def process_dataset(dataset, vocab_table, sos, eos, batch_size=8, num_epochs=1, num_parallel_calls=32, is_infer=False):

    output_buffer_size = batch_size * 1000

    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

    dataset = dataset.repeat(num_epochs)

    if not is_infer:
        dataset = dataset.shuffle(output_buffer_size)

    dataset = dataset.map(
        lambda inputs, labels: (inputs,
                                vocab_table.lookup(labels)),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(
        lambda inputs, labels: (tf.cast(inputs, tf.float32),
                                tf.cast(labels, tf.int32)),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(
        lambda inputs, labels: (inputs,
                                tf.concat(([sos_id], labels), 0),
                                tf.concat((labels, [eos_id]), 0)),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(
        lambda inputs, labels_in, labels_out: (inputs,
                                               labels_in,
                                               labels_out,
                                               tf.shape(inputs)[0],
                                               tf.size(labels_in)),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(
        lambda inputs, labels_in, labels_out,
        source_sequence_length, target_sequence_length: (
            {
                'encoder_inputs': inputs,
                'source_sequence_length': source_sequence_length,
            },
            {
                'targets_inputs': labels_in,
                'targets_outputs': labels_out,
                'target_sequence_length': target_sequence_length
            }))

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=dataset.output_shapes,
        padding_values=(
            {
                'encoder_inputs': 0.0,
                'source_sequence_length': 0,
            },
            {
                'targets_inputs': eos_id,
                'targets_outputs': eos_id,
                'target_sequence_length': 0,
            }))

    '''
    dataset = dataset.filter(lambda features, labels: tf.equal(
        tf.shape(features['source_sequence_length'])[0], batch_size))
    '''

    return dataset
