import sys

sys.path.append("net")
sys.path.append("net/modelzoo")
sys.path.append("net/basemodel")
sys.path.append("motion")

import torch
from absl import app
from absl import flags
from torch.utils.data import DataLoader
import pandas as pd

from BodyAE import BodyAE
from BodyMotionGenerator import BodyMotionGenerator
from ConvMotionTransformVAE import ConvMotionTransformVAE
from LstmBodyAE import LstmBodyAE
from HagglingDataset import HagglingDataset
from CharControlMotionVAE import CharControlMotionVAE
from Metrics import Metrics

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 256, 'Training set mini batch size')
flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('test', 'MannData/test/', 'Directory containing test files')
flags.DEFINE_string('output_dir', 'Data/output/', 'Folder to store final videos')
flags.DEFINE_string('ckpt', 'ckpt/', 'file containing the model weights')
flags.DEFINE_float('lmd', 0.2, 'L1 Regularization factor')
flags.DEFINE_boolean('bodyae', False, 'if True checks BodyAE model')
flags.DEFINE_integer('enc_hidden_units', 256, 'Encoder LSTM hidden units')
flags.DEFINE_integer('dec_hidden_units', 256, 'Decoder LSTM hidden units')
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
flags.DEFINE_integer('c_dim', 2, 'number of conditional variables added to latent dimension')
flags.DEFINE_bool('speak', True, 'speak classification required')
flags.DEFINE_float('lmd2', 0.2, 'Regularization factor for speaking predcition')
flags.DEFINE_integer('frechet_pose_dim', 42, 'Number of joint directions')
flags.DEFINE_string('frechet_ckpt', 'ckpt/Frechet/', 'file containing the model weights')

flags.DEFINE_integer('input_dim', 244, 'input pose vector dimension')
flags.DEFINE_integer('output_dim', 244, 'input pose vector dimension')
flags.DEFINE_string('model', "MVAE", 'Defines the name of the model')
flags.DEFINE_bool('pretrain', False, 'Use a pretrained model')
flags.DEFINE_bool('CNN', False, 'Cnn based model')
flags.DEFINE_bool('VAE', True, 'VAE training')
flags.DEFINE_string('pretrainedModel', 'bodyAE', 'path to pretrained weights')
flags.DEFINE_integer('batch_runs', 5, 'Number of times give the same input to VAE')
flags.DEFINE_integer('num_saves', 10, 'number of outputs to save')
flags.DEFINE_integer('test_ckpt', None, 'checkpoint to test')
flags.DEFINE_string('fmt', 'mann', 'data format')
flags.DEFINE_string('device', 'cuda:0', 'Device to train on')

pss = lambda a, b: a == b


def get_model():
    """
    Returns the appropriate model for training
    :return: PyTorch model that extends nn.Module
    """
    if FLAGS.model == 'bodyAE':
        x = BodyAE(FLAGS).to(torch.device(FLAGS.device))
        return x
    elif FLAGS.model == 'lstmAE':
        return LstmBodyAE(FLAGS).to(torch.device(FLAGS.device))
    elif FLAGS.model == 'MTVAE':
        return ConvMotionTransformVAE(FLAGS).to(torch.device(FLAGS.device))
    elif FLAGS.model == 'MVAE':
        return CharControlMotionVAE(FLAGS).to(torch.device(FLAGS.device))
    else:
        return BodyMotionGenerator(FLAGS).to(torch.device(FLAGS.device))


def main(arg):
    test_dataset = HagglingDataset(FLAGS.test, FLAGS)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, num_workers=10)

    ckpt = FLAGS.ckpt + FLAGS.model

    model = get_model()
    model.load_model(ckpt, FLAGS.test_ckpt)
    model.eval()

    metrics = Metrics(FLAGS)

    df = pd.DataFrame()

    with torch.no_grad():
        for i_batch, batch in enumerate(test_dataloader):

            batch_runs = FLAGS.batch_runs
            if FLAGS.VAE:
                batch_runs = FLAGS.batch_runs

            for test_num in range(0, batch_runs):
                predictions, targets = model(batch)

                out = metrics.compute_and_save(predictions, targets, batch, i_batch, test_num)

                print(out)

                df = df.append(out, ignore_index=True)

        df_mean = df.mean(axis=0)
        df_std = df.std(axis=0)
        print(df_mean)
        print(df_std)
        df_mean.to_csv('testResults/'+FLAGS.model+'/mean.csv')
        df_std.to_csv('testResults/' + FLAGS.model + '/std.csv')


if __name__ == "__main__":
    app.run(main)
