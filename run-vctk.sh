ROOT_DIR=$(mktemp -d)

VCTK_TAR_FILE=$ROOT_DIR/VCTK-Corpus.tar.gz
VCTK_DIR=$ROOT_DIR/VCTK-Corpus
VCTK_PROCESSED_DIR=$VCTK_DIR/processed
VCTK_TF_DIR=$VCTK_DIR/tfdata
VCTK_MODEL_DIR=$VCTK_DIR/model

VCTK_URL="http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"

REPO_VCTK_DIR=./vctk

echo All data and model will be saved at $ROOT_DIR

if [ ! -e $VCTK_TAR_FILE ]
then
    echo Downloading dataset ...
    wget $VCTK_URL -O $VCTK_TAR_FILE
fi

echo Extracting files ...
tar -zxvf $VCTK_TAR_FILE -C $ROOT_DIR

mkdir -p $VCTK_PROCESSED_DIR

echo Processing dataset ...
python3 $REPO_VCTK_DIR/preprocess.py --data_dir $VCTK_DIR --output_dir $VCTK_PROCESSED_DIR

mkdir -p $VCTK_TF_DIR

for data in train test
do
    echo Writing $data TFRecord ...
    python3 $REPO_VCTK_DIR/write_tfrecord.py --inputs $VCTK_PROCESSED_DIR/$data.feat.npy \
                                             --labels $VCTK_PROCESSED_DIR/$data.label.npy \
                                             --output $VCTK_TF_DIR/$data.tfrecord
done

echo Building Vocab table
python3 $REPO_VCTK_DIR/build_vocab.py $VCTK_PROCESSED_DIR/train.label.npy $VCTK_TF_DIR/vocab.table

echo Training ...
python3 train.py --train $VCTK_TF_DIR/train.tfrecord \
                 --valid $VCTK_TF_DIR/test.tfrecord \
                 --vocab $VCTK_TF_DIR/vocab.table \
                 --model_dir $VCTK_MODEL_DIR \
                 --encoder_layers 3 \
                 --encoder_units 128 \
                 --decoder_layers 1 \
                 --decoder_units 128 \
                 --dropout 0.2 \
                 --batch_size 32 \
                 --use_pyramidal \
                 --embedding_size 0 \
                 --sampling_probability 0.2 \
                 --eval_secs 1200 \
                 --attention_type luong
