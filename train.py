import os
import argparse
import tensorflow as tf

import utils

from model_helper import las_model_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Listen, Attend and Spell(LAS) implementation based on Tensorflow. '
                    'The model utilizes input pipeline and estimator API of Tensorflow, '
                    'which makes the training procedure truly end-to-end.')

    parser.add_argument('--train', type=str, required=True,
                        help='training data in TFRecord format')
    parser.add_argument('--valid', type=str,
                        help='validation data in TFRecord format')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--mapping', type=str,
                        help='additional mapping when evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of saving model')
    parser.add_argument('--eval_secs', type=int, default=300,
                        help='evaluation every N seconds, only happening when `valid` is specified')

    parser.add_argument('--encoder_units', type=int, default=128,
                        help='rnn hidden units of encoder')
    parser.add_argument('--encoder_layers', type=int, default=3,
                        help='rnn layers of encoder')
    parser.add_argument('--use_pyramidal', action='store_true',
                        help='whether to use pyramidal rnn')

    parser.add_argument('--decoder_units', type=int, default=128,
                        help='rnn hidden units of decoder')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='rnn layers of decoder')
    parser.add_argument('--embedding_size', type=int, default=0,
                        help='embedding size of target vocabulary, if 0, one hot encoding is applied')
    parser.add_argument('--sampling_probability', type=float, default=0.1,
                        help='sampling probabilty of decoder during training')
    parser.add_argument('--attention_type', type=str, default='luong', choices=['luong', 'custom'],
                        help='type of attention mechanism')
    parser.add_argument('--attention_layer_size', type=int,
                        help='size of attention layer, see tensorflow.contrib.seq2seq.AttentionWrapper'
                             'for more details')
    parser.add_argument('--bottom_only', action='store_true',
                        help='apply attention mechanism only at the bottommost rnn cell')
    parser.add_argument('--pass_hidden_state', action='store_true',
                        help='whether to pass encoder state to decoder')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate of rnn cell')

    return parser.parse_args()


def input_fn(dataset_filename, vocab_filename, num_channels=39, batch_size=8, num_epochs=1):
    dataset = utils.read_dataset(dataset_filename, num_channels)
    vocab_table = utils.create_vocab_table(vocab_filename)

    dataset = utils.process_dataset(
        dataset, vocab_table, utils.SOS, utils.EOS, batch_size, num_epochs)

    return dataset


def main(args):

    vocab_list = utils.load_vocab(args.vocab)

    vocab_size = len(vocab_list)

    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(
        args, vocab_size, utils.SOS_ID, utils.EOS_ID)

    model = tf.estimator.Estimator(
        model_fn=las_model_fn,
        config=config,
        params=hparams)

    if args.valid:
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(
                args.train, args.vocab, num_channels=args.num_channels, batch_size=args.batch_size, num_epochs=args.num_epochs))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(
                args.valid or args.train, args.vocab, num_channels=args.num_channels, batch_size=args.batch_size),
            start_delay_secs=60,
            throttle_secs=args.eval_secs)

        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        model.train(
            input_fn=lambda: input_fn(
                args.train, args.vocab, num_channels=args.num_channels, batch_size=args.batch_size, num_epochs=args.num_epochs))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    args = parse_args()

    main(args)
