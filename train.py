import sys
import wandb

from Metrics import Metrics
from net.modelzoo.BodyAE import BodyAE
from net.modelzoo.BodyMotionGenerator import BodyMotionGenerator
from net.modelzoo.CharControlMotionVAE import CharControlMotionVAE
from net.modelzoo.ConvMotionTransformVAE import ConvMotionTransformVAE
from net.modelzoo.LstmBodyAE import LstmBodyAE
from net.modelzoo.LSTMMotionTransformVAE import LSTMMotionTransformVAE

sys.path.append("net")
sys.path.append("net/modelzoo")
sys.path.append("net/basemodel")
sys.path.append("motion")

import os
from torch.utils.data import DataLoader

from absl import app
from absl import flags

from HagglingDataset import HagglingDataset

from net.losses import *

FLAGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('train', 'MannData/train/', 'Directory containing train files')
flags.DEFINE_string('test', 'MannData/test/', 'Directory containing train files')
flags.DEFINE_string('ckpt_dir', 'ckpt/', 'Directory to store checkpoints')
flags.DEFINE_string('frechet_ckpt', 'ckpt/Frechet/', 'file containing the model weights')
flags.DEFINE_string('output_dir', 'Data/MVAEoutput/', 'Folder to store final videos')

flags.DEFINE_integer('batch_size', 48, 'Training set mini batch size')
flags.DEFINE_integer('epochs', 400, 'Training epochs')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
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
flags.DEFINE_integer('c_dim', 0, 'number of conditional variables added to latent dimension')
flags.DEFINE_bool('speak', True, 'speak classification required')
flags.DEFINE_float('lmd2', 0.2, 'Regularization factor for speaking predcition')
flags.DEFINE_float('lmd3', 0.2, 'Regularization factor for velocity predcition')
flags.DEFINE_bool('skip_train_metrics', True, 'skip calculation of train metrics')

flags.DEFINE_integer('input_dim', 244, 'input pose vector dimension')
flags.DEFINE_integer('output_dim', 244, 'output pose vector dimension')
flags.DEFINE_bool('pretrain', True, 'pretrain the auto encoder')
flags.DEFINE_bool('resume_train', False, 'Resume training the model')
flags.DEFINE_string('model', "MVAE", 'Defines the name of the model')
flags.DEFINE_bool('CNN', False, 'Cnn based model')
flags.DEFINE_string('pretrainedModel', 'bodyAE', 'path to pretrained weights')
flags.DEFINE_integer('ckpt', 10, 'Number of epochs to checkpoint')
flags.DEFINE_integer('pretrained_ckpt', None, 'Number of epochs to checkpoint of pretrained model')
flags.DEFINE_string('device', 'cuda:0', 'Device to train on')
flags.DEFINE_integer('num_saves', 0, 'number of output videos to save')
flags.DEFINE_string('fmt', 'mann', 'data format')
flags.DEFINE_integer('frechet_pose_dim', 42, 'Number of joint directions')


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
    elif FLAGS.model == 'LMTVAE':
        return LSTMMotionTransformVAE(FLAGS).to(torch.device(FLAGS.device))
    elif FLAGS.model == 'MVAE':
        return CharControlMotionVAE(FLAGS).to(torch.device(FLAGS.device))
    else:
        return BodyMotionGenerator(FLAGS).to(torch.device(FLAGS.device))


def get_loss_fn():
    """
    Returns the appropriate loss function for training
    :return: loss function
    """
    if FLAGS.model == "MTVAE" or FLAGS.model == 'LMTVAE':
        return reconstruction_VAE
    elif FLAGS.model == 'MVAE':
        return sequential_reconstruction_VAE
    else:
        return reconstruction_l1


def get_optimizer(parameters):
    """
    Returns the appropriate optimizer
    :parameters: model parametres to be tuned
    :return: optimizer
    """
    if FLAGS.optimizer == "adam":
        return torch.optim.Adamax(parameters, lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=FLAGS.learning_rate)
    else:
        return torch.optim.SGD(parameters, lr=FLAGS.learning_rate)


def get_scheduler(optimizer):
    """
    Returns the LR scheduler
    :param optimizer: Pytorch optimizerbodyAE
    :return: pytorch lr scheduler
    """

    return torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)


def decay_p(p, epoch, model):
    """
    Returns the current value of p according to the progress in training
    :param p : degree of teacher forcing
    :param epoch: current epoch
    """
    if FLAGS.start_scheduled_sampling < epoch / FLAGS.epochs < FLAGS.end_scheduled_sampling:
        num_epochs_decay = (FLAGS.end_scheduled_sampling - FLAGS.start_scheduled_sampling) * FLAGS.epochs
        p -= 1 / num_epochs_decay
    elif epoch / FLAGS.epochs > FLAGS.end_scheduled_sampling:
        p = 0

    if FLAGS.model == 'MVAE' or FLAGS.model == 'LMTVAE':
        model.set_teacher_forcing(p)

    return


def main(args):
    # make sure dec hidden units and layers are same
    FLAGS.dec_hidden_units = FLAGS.enc_hidden_units
    FLAGS.dec_layers = FLAGS.enc_layers

    # initialize the dataset and the data loader
    train_dataset = HagglingDataset(FLAGS.train, FLAGS)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=10)
    test_dataset = HagglingDataset(FLAGS.test, FLAGS)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=10)

    # set the wandb config
    config = FLAGS.flag_values_dict()
    run = wandb.init(project="Sell-It", config=config)

    # initialize the model, log it for visualization
    model = get_model()
    # try:
    #     torch.onnx.export(model, next(iter(train_dataloader)),
    #                       os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/model.onnx'))
    #     wandb.save(os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/model.onnx'))
    # except Exception as e:
    #     print(e)

    starting_epoch = 0

    # restore model if needed
    if FLAGS.resume_train:
        ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/')
        starting_epoch = model.load_model(ckpt, None)
        starting_epoch = 90

    # get the loss function and optimizers
    criterion = get_loss_fn()
    optimizer = get_optimizer(model.get_trainable_parameters())
    p = 1.0

    metrics = Metrics(FLAGS)

    # run the training script
    for epoch in range(starting_epoch + 1, FLAGS.epochs + 1):

        print(epoch)

        # initialize the total epoch loss values
        train_loss_logs = {
            'Train/Total_Loss': 0,
            'Train/Reconstruction_Loss': 0,
            'Train/Regularization_Loss': 0,
            'Train/CrossEntropy_Loss': 0,
            'Train/VelocityRegularization': 0
        }

        train_metric_logs = {
            'Train/RightMSE': 0,
            'Train/LeftMSE': 0,
            'Train/RightNPSS': 0,
            'Train/LeftNPSS': 0,
            'Train/RightFrechet': 0,
            'Train/LeftFrechet': 0,
            'Train/RightSpeech': 0,
            'Train/LeftSpeech': 0,
            'Train/MSE': 0,
            'Train/NPSS': 0,
            'Train/Frechet': 0,
            'Train/Speech': 0

        }

        test_metric_logs = {
            'Test/RightMSE': 0,
            'Test/LeftMSE': 0,
            'Test/RightNPSS': 0,
            'Test/LeftNPSS': 0,
            'Test/RightFrechet': 0,
            'Test/LeftFrechet': 0,
            'Test/RightSpeech': 0,
            'Test/LeftSpeech': 0,
            'Test/MSE': 0,
            'Test/NPSS': 0,
            'Test/Frechet': 0,
            'Test/Speech': 0
        }

        # set model to train mode
        model.train()

        # decay factor set
        decay_p(p, epoch, model)

        # run through all the batches
        for i_batch, batch in enumerate(train_dataloader):
            # zero prev gradients
            optimizer.zero_grad()

            # forward pass through the net
            predictions, targets = model(batch)

            # calculate loss
            losses = criterion(predictions, targets, model.parameters(), FLAGS)
            total_loss = losses['Total_Loss']

            # calculate gradients
            total_loss.backward()
            optimizer.step()

            # compute train metrics
            with torch.no_grad():
                if not FLAGS.skip_train_metrics:
                    train_metrics = metrics.compute_and_save(predictions, targets, batch, i_batch, None)
                    train_metric_logs = {
                        'Train/' + key: train_metrics[key] + train_metric_logs['Train/' + key] for key in train_metrics
                    }

                train_loss_logs = {
                    'Train/' + key: losses[key].detach().cpu().numpy().item() + train_loss_logs['Train/' + key] for key
                    in losses
                }

        # set the model to evaluation mode
        model.eval()

        # calculate validation loss
        with torch.no_grad():
            for i_batch, batch in enumerate(test_dataloader):
                # forward pass through the net
                predictions, targets = model(batch)

                if FLAGS.model == 'bodyAE' or FLAGS.model == 'bmg':
                    test_metric_logs['Test/MSE'] += meanJointPoseError(predictions, targets)
                else:
                    # consolidate metrics
                    test_metrics = metrics.compute_and_save(predictions, targets, batch, i_batch, None)
                    test_metric_logs = {
                        'Test/' + key: test_metrics[key] + test_metric_logs['Test/' + key] for key in test_metrics
                    }

        # scale the metrics
        train_metric_logs = {
            key: train_metric_logs[key] / len(train_dataloader) for key in train_metric_logs
        }
        train_loss_logs = {
            key: train_loss_logs[key] / len(train_dataloader) for key in train_loss_logs
        }
        test_metric_logs = {
            key: test_metric_logs[key] / len(test_dataloader) for key in test_metric_logs
        }

        # log all the metrics
        run.log({
            **train_metric_logs,
            **train_loss_logs,
            **test_metric_logs
        })

        if epoch % FLAGS.ckpt == 0 and epoch > 0:
            ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/' + wandb.run.name + '/')
            model.save_model(ckpt, epoch)

    run.finish()


if __name__ == '__main__':
    app.run(main)
