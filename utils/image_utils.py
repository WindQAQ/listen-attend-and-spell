import tensorflow as tf

__all__ = [
    'create_attention_images',
]


"""Reference: https://github.com/tensorflow/nmt/blob/master/nmt/attention_model.py"""


def create_attention_images(final_context_state):
    attention_images = (final_context_state.alignment_history.stack())
    # Reshape to (batch, src_seq_len, tgt_seq_len,1)
    attention_images = tf.expand_dims(
        tf.transpose(attention_images, [1, 2, 0]), -1)
    # Scale to range [0, 255]
    attention_images *= 255

    return attention_images
