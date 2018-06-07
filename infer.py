import argparse
import numpy as np
import tensorflow as tf

import utils

from model_helper import las_model_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Listen, Attend and Spell(LAS) implementation based on Tensorflow. '
                    'The model utilizes input pipeline and estimator API of Tensorflow, '
                    'which makes the training procedure truly end-to-end.')

    parser.add_argument('--data', type=str, required=True,
                        help='inference data in TFRecord format')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of imported model')
    parser.add_argument('--save', type=str, required=True,
                        help='path of saving inference results')

    parser.add_argument('--beam_width', type=int, default=0,
                        help='number of beams (default 0: using greedy decoding)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')

    return parser.parse_args()


def input_fn(dataset_filename, vocab_filename, num_channels=39, batch_size=8, num_epochs=1):
    dataset = utils.read_dataset(dataset_filename, num_channels)
    vocab_table = utils.create_vocab_table(vocab_filename)

    dataset = utils.process_dataset(
        dataset, vocab_table, utils.SOS, utils.EOS, batch_size, num_epochs)

    return dataset


def main(args):

    vocab_list = np.array(utils.load_vocab(args.vocab))

    vocab_size = len(vocab_list)

    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(
        args, vocab_size, utils.SOS_ID, utils.EOS_ID)

    model = tf.estimator.Estimator(
        model_fn=las_model_fn,
        config=config,
        params=hparams)

    predictions = model.predict(
        input_fn=lambda: input_fn(
            args.data, args.vocab, num_channels=args.num_channels, batch_size=args.batch_size, num_epochs=1),
        predict_keys='sample_ids')

    predictions = [vocab_list[y['sample_ids']].tolist() + [utils.EOS]
                   for y in predictions]

    predictions = [' '.join(y[:y.index(utils.EOS)]) for y in predictions]

    with open(args.save, 'w') as f:
        f.write('\n'.join(predictions))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    args = parse_args()

    main(args)
