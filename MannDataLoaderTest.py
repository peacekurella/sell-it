import sys

sys.path.append("net")
sys.path.append("net/modelzoo")
sys.path.append("net/basemodel")
sys.path.append("motion")

import os
import wandb
from torch.utils.data import DataLoader

from absl import app
from absl import flags

from MannDataset import MannDataset
from BodyAE import BodyAE
from BodyMotionGenerator import BodyMotionGenerator
from LstmBodyAE import LstmBodyAE
from ConvMotionTransformVAE import ConvMotionTransformVAE
from CharControlMotionVAE import CharControlMotionVAE

from losses import *

FLAGS = flags.FLAGS

flags.DEFINE_string('meta', 'MannData/stats/', 'Directory containing metadata files')
flags.DEFINE_string('train', 'MannData/train/', 'Directory containing train files')
flags.DEFINE_string('test', 'MannData/test/', 'Directory containing train files')
flags.DEFINE_string('ckpt_dir', 'ckpt/', 'Directory to store checkpoints')

flags.DEFINE_integer('batch_size', 16, 'Training set mini batch size')
flags.DEFINE_integer('epochs', 80, 'Training epochs')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('lmd', 0.2, 'Regularization factor')
flags.DEFINE_string('optimizer', 'Adam', 'type of optimizer')
flags.DEFINE_integer('enc_hidden_units', 256, 'Encoder hidden units')
flags.DEFINE_integer('dec_hidden_units', 256, 'Decoder hidden units')
flags.DEFINE_integer('gat_hidden_units', 256, 'Gating network hidden units')
flags.DEFINE_integer('enc_layers', 3, 'encoder layers')
flags.DEFINE_integer('dec_layers', 1, 'decoder layers')
flags.DEFINE_integer('num_experts', 8, 'number of experts in  decoder')
flags.DEFINE_float('enc_dropout', 0.25, 'encoder dropout')
flags.DEFINE_float('dec_dropout', 0.25, 'decoder dropout')
flags.DEFINE_float('gat_dropout', 0.25, 'gating network dropout')
flags.DEFINE_float('dropout', 0.25, 'dense network dropout')
flags.DEFINE_float('tf_ratio', 0.3, 'teacher forcing ratio')
flags.DEFINE_integer('seq_length', 120, 'time steps in the sequence')
flags.DEFINE_integer('latent_dim', 32, 'latent dimension')
flags.DEFINE_float('start_scheduled_sampling', 0.2, 'when to start scheduled sampling')
flags.DEFINE_float('end_scheduled_sampling', 0.4, 'when to stop scheduled sampling')

flags.DEFINE_integer('input_dim', 244, 'input pose vector dimension')
flags.DEFINE_integer('output_dim', 244, 'input pose vector dimension')
flags.DEFINE_bool('pretrain', False, 'pretrain the auto encoder')
flags.DEFINE_bool('resume_train', False, 'Resume training the model')
flags.DEFINE_string('model', "MVAE", 'Defines the name of the model')
flags.DEFINE_bool('CNN', False, 'Cnn based model')
flags.DEFINE_bool('VAE', True, 'VAE training')
flags.DEFINE_string('pretrainedModel', 'bodyAE', 'path to pretrained weights')
flags.DEFINE_integer('ckpt', 10, 'Number of epochs to checkpoint')


def main(args):
    # make sure dec hidden units and layers are same
    FLAGS.dec_hidden_units = FLAGS.enc_hidden_units
    FLAGS.dec_layers = FLAGS.enc_layers

    # initialize the dataset and the data loader
    train_dataset = MannDataset(FLAGS.train, FLAGS)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=10)
    test_dataset = MannDataset(FLAGS.test, FLAGS)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=10)

    dataiter = iter(train_dataloader)
    print(dataiter.next().keys())

if __name__ == '__main__':
    app.run(main)