import sys

sys.path.append("net")
sys.path.append("net/modelzoo")
sys.path.append("net/basemodel")
sys.path.append("motion")

import torch
from absl import app
from absl import flags
from torch.utils.data import DataLoader

from BodyAE import BodyAE
from BodyMotionGenerator import BodyMotionGenerator
from ConvMotionTransformVAE import ConvMotionTransformVAE
from LstmBodyAE import LstmBodyAE
from HagglingDataset import HagglingDataset
from CharControlMotionVAE import CharControlMotionVAE
from Metrics import Metrics

FLAGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('test', 'MannData/test/', 'Directory containing test files')
flags.DEFINE_string('output_dir', 'Data/MVAEHoutput/', 'Folder to store final videos')
flags.DEFINE_string('ckpt', 'ckpt/MVAE/ME', 'file containing the model weights')
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

flags.DEFINE_integer('input_dim', 244, 'input pose vector dimension')
flags.DEFINE_integer('output_dim', 244, 'input pose vector dimension')
flags.DEFINE_string('model', "MVAE", 'Defines the name of the model')
flags.DEFINE_bool('pretrain', False, 'Use a pretrained model')
flags.DEFINE_bool('CNN', False, 'Cnn based model')
flags.DEFINE_bool('VAE', True, 'VAE training')
flags.DEFINE_string('pretrainedModel', 'bodyAE', 'path to pretrained weights')
flags.DEFINE_integer('batch_runs', 5, 'Number of times give the same input to VAE')
flags.DEFINE_integer('num_saves', 20, 'number of outputs to save')
flags.DEFINE_integer('test_ckpt', None, 'checkpoint to test')
flags.DEFINE_string('fmt', 'mann', 'data format')

pss = lambda a, b: a == b


def get_input(batch):
    """
    Generates the inputs to the network
    :param batch: batch for training
    :return: train_x, train_y tensors
    """
    b = batch['buyer']['joints21']
    l = batch['leftSeller']['joints21']
    r = batch['rightSeller']['joints21']

    if FLAGS.pretrain:
        if FLAGS.CNN:
            train_x = torch.cat((l, r), dim=0).permute(0, 2, 1).float().cuda()
            train_y = torch.cat((l, r), dim=0).permute(0, 2, 1).float().cuda()
            return train_x, train_y
        else:
            train_x = torch.cat((l, r), dim=0).float().cuda()
            train_y = torch.cat((l, r), dim=0).float().cuda()
            return train_x, train_y
    else:
        if FLAGS.CNN:
            set_x_a = torch.cat((b, l), dim=2)
            set_x_b = torch.cat((b, r), dim=2)
            train_x = torch.cat((set_x_a, set_x_b), dim=0).permute(0, 2, 1).float().cuda()
            train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float().cuda()
            return train_x, train_y
        else:
            set_x_a = torch.cat((b, l), dim=2)
            set_x_b = torch.cat((b, r), dim=2)
            train_x = torch.cat((set_x_a, set_x_b), dim=0).float().cuda()
            train_y = torch.cat((r, l), dim=0).float().cuda()
            return train_x, train_y


def get_model():
    """
    Returns the appropriate model for training
    :return: PyTorch model that extends nn.Module
    """

    if FLAGS.model == 'bodyAE':
        return BodyAE(FLAGS).cuda()
    if FLAGS.model == 'LstmAE':
        return LstmBodyAE(FLAGS).cuda()
    if FLAGS.model == 'MTVAE':
        return ConvMotionTransformVAE(FLAGS).cuda()
    if FLAGS.model == 'MVAE':
        return CharControlMotionVAE(FLAGS).cuda()
    else:
        return BodyMotionGenerator(FLAGS).cuda()


def main(arg):

    test_dataset = HagglingDataset(FLAGS.test, FLAGS)
    test_dataloader = DataLoader(test_dataset, num_workers=10)

    ckpt = FLAGS.ckpt

    model = get_model()
    model.load_model(ckpt, FLAGS.test_ckpt)
    model.eval()

    metrics = Metrics(FLAGS)

    with torch.no_grad():
        for i_batch, batch in enumerate(test_dataloader):

            batch_runs = 1
            if FLAGS.VAE:
                batch_runs = FLAGS.batch_runs

            for test_num in range(0, batch_runs):

                # get test input and labels
                data, targets = get_input(batch)

                # forward pass through the network
                if FLAGS.VAE:
                    data = (data, targets)
                    predictions = model(data, 0)
                else:
                    predictions = model(data)

                if FLAGS.CNN:
                    targets = targets.permute(0, 2, 1)
                    predictions = predictions.permute(0, 2, 1)

                out = metrics.compute_and_save(predictions, targets, batch, i_batch, test_num)
                print(out)

if __name__ == "__main__":
    app.run(main)
