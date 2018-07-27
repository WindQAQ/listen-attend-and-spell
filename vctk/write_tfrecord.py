import numpy as np
import tensorflow as tf
from argparse import ArgumentParser


def make_example(input, label):
    feature_lists = tf.train.FeatureLists(feature_list={
        'labels': tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[p.encode()]))
            for p in label
        ]),
        'inputs': tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=f))
            for f in input
        ])
    })

    return tf.train.SequenceExample(feature_lists=feature_lists)


parser = ArgumentParser()
parser.add_argument('--inputs')
parser.add_argument('--labels')
parser.add_argument('--output')

args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.output) as writer:
    inputs = np.load(args.inputs).item()
    labels = np.load(args.labels).item()

    assert len(inputs) == len(labels)

    for i, (name, input) in enumerate(inputs.items()):
        label = labels[name]

        if i < 10:
            print(name, label)

        writer.write(make_example(input, label).SerializeToString())
