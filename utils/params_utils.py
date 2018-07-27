import os
import json
import tensorflow as tf

__all__ = [
    'HParams',
    'create_hparams',
]


class HParams(tf.contrib.training.HParams):
    def del_hparam(self, name):
        if hasattr(self, name):
            delattr(self, name)
            del self._hparam_types[name]

    def pop_hparam(self, name):
        value = getattr(self, name)
        self.del_hparam(name)

        return value

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f)


def get_default_hparams():
    hparams = HParams(
        # global setting
        learning_rate=1e-3,
        dropout=0.2,

        # encoder setting
        encoder_layers=3,
        encoder_units=64,
        use_pyramidal=True,

        # decoder setting
        decoder_layers=2,
        decoder_units=128,
        target_vocab_size=0,
        embedding_size=0,
        sampling_probability=0.1,
        sos_id=1,
        eos_id=2,
        bottom_only=False,
        pass_hidden_state=False,
        decoding_length_factor=1.0,
        attention_type='luong',
        attention_layer_size=None,
        beam_width=0,

        # evaluation setting
        mapping=None)

    return hparams


def create_hparams(args, target_vocab_size, sos_id=1, eos_id=2):
    hparams = get_default_hparams()
    hparams_file = os.path.join(args.model_dir, 'hparams.json')

    if os.path.exists(hparams_file):
        with open(hparams_file, 'r') as f:
            hparams_dict = json.loads(json.load(f))

        for name, value in vars(args).items():
            if name not in hparams_dict:
                hparams_dict[name] = value
    else:
        hparams_dict = {
            **vars(args),
            **{'sos_id': sos_id, 'eos_id': eos_id, 'target_vocab_size': target_vocab_size},
        }

    for name, value in hparams.values().items():
        value = hparams_dict.get(name, None)
        if value is not None:
            if name == 'mapping':
                if not isinstance(value, list):
                    value = [int(x.strip()) for x in open(value, 'r')]
                hparams.del_hparam(name)
                hparams.add_hparam(name, value)
            else:
                hparams.set_hparam(name, value)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    hparams.save_to_file(hparams_file)

    return get_encoder_decoder_hparams(hparams)


def get_encoder_decoder_hparams(hparams):
    learning_rate = hparams.pop_hparam('learning_rate')
    dropout = hparams.pop_hparam('dropout')
    mapping = hparams.pop_hparam('mapping')

    encoder_hparams = HParams(
        num_layers=hparams.pop_hparam('encoder_layers'),
        num_units=hparams.pop_hparam('encoder_units'),
        use_pyramidal=hparams.pop_hparam('use_pyramidal'),
        dropout=dropout)

    decoder_hparams = HParams(
        num_layers=hparams.pop_hparam('decoder_layers'),
        num_units=hparams.pop_hparam('decoder_units'),
        dropout=dropout)

    for name, value in hparams.values().items():
        decoder_hparams.add_hparam(name, value)

    return HParams(
        learning_rate=learning_rate,
        mapping=mapping,
        encoder=encoder_hparams,
        decoder=decoder_hparams)
