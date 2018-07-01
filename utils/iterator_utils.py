import tensorflow as tf

__all__ = [
    'get_iterator',
]


def get_iterator(dataset, vocab_table, sos, eos, batch_size=8, num_parallel_calls=32, random_seed=42):

    output_buffer_size = batch_size * 1000

    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

    dataset = dataset.shuffle(output_buffer_size, random_seed)

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

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=dataset.output_shapes,
        padding_values=(0.0, eos_id, eos_id, 0, 0))

    return dataset
    # return dataset.make_initializable_iterator()
