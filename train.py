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

from HagglingDataset import HagglingDataset
from BodyAE import BodyAE
from BodyMotionGenerator import BodyMotionGenerator
from LstmBodyAE import LstmBodyAE

from losses import *

FLAGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('train', 'Data/train/', 'Directory containing train files')
flags.DEFINE_string('test', 'Data/test/', 'Directory containing train files')
flags.DEFINE_string('ckpt_dir', 'ckpt/', 'Directory to store checkpoints')

flags.DEFINE_integer('batch_size', 64, 'Training set mini batch size')
flags.DEFINE_integer('epochs', 150, 'Training epochs')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_float('lmd', 0.000, 'L1 Regularization factor')
flags.DEFINE_string('optimizer', 'Adam', 'type of optimizer')
flags.DEFINE_integer('enc_hidden_units', 128, 'Encoder LSTM hidden units')
flags.DEFINE_integer('dec_hidden_units', 128, 'Decoder LSTM hidden units')
flags.DEFINE_integer('enc_layers', 1, 'encoder LSTM layers')
flags.DEFINE_integer('dec_layers', 1, 'decoder LSTM layers')
flags.DEFINE_float('enc_dropout', 0.25, 'encoder LSTM dropout')
flags.DEFINE_float('dec_dropout', 0.25, 'decoder LSTM dropout')
flags.DEFINE_float('dropout', 0.25, 'dense network dropout')
flags.DEFINE_float('tf_ratio', 0.5, 'teacher forcing ratio')

flags.DEFINE_integer('input_dim', 73, 'input pose vector dimension')
flags.DEFINE_integer('output_dim', 73, 'input pose vector dimension')
flags.DEFINE_bool('pretrain', True, 'pretrain the auto encoder')
flags.DEFINE_bool('resume_train', False, 'Resume training the model')
flags.DEFINE_string('model', "LstmAE", 'Defines the name of the model')
flags.DEFINE_bool('CNN', False, 'Cnn based model')
flags.DEFINE_integer('ckpt', 10, 'Number of epochs to checkpoint')


def get_inputs(batch):
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
    else:
        return BodyMotionGenerator(FLAGS).cuda()


def get_loss_fn():
    """
    Returns the appropriate loss function for training
    :return: loss function
    """

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
    :param optimizer: Pytorch optimizer
    :return: pytorch lr scheduler
    """

    return torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)


def get_hyperparameters():
    if FLAGS.model == "bodyAE":
        hyperparameter_defaults = dict(
            batch_size=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            epochs=FLAGS.epochs,
            lmd=FLAGS.lmd,
            optimizer=FLAGS.optimizer
        )
        return hyperparameter_defaults
    else:
        hyperparameter_defaults = dict(
            batch_size=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            epochs=FLAGS.epochs,
            lmd=FLAGS.lmd,
            optimizer=FLAGS.optimizer,
            enc_hidden_units=FLAGS.enc_hidden_units,
            dec_hidden_units=FLAGS.dec_hidden_units,
            enc_layers=FLAGS.enc_layers,
            dec_layers=FLAGS.dec_layers,
            dropout=FLAGS.dropout,
            tf_ratio=FLAGS.tf_ratio
        )
        return hyperparameter_defaults


def main(args):
    # make sure dec hidden units and layers are same
    FLAGS.dec_hidden_units = FLAGS.enc_hidden_units
    FLAGS.dec_layers = FLAGS.enc_layers

    # initialize the dataset and the data loader
    train_dataset = HagglingDataset(FLAGS.train, FLAGS)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=10)
    test_dataset = HagglingDataset(FLAGS.test, FLAGS)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=10)

    # get default_hyperparameters
    hyperparameter_defaults = get_hyperparameters()

    # initialize the model
    model = get_model()

    run = wandb.init(project=FLAGS.model, config=hyperparameter_defaults)

    # restore model if needed
    if FLAGS.resume_train:
        if FLAGS.pretrain:
            ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/AE/')
        else:
            ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/ME/')
        model.load_model(ckpt, None)

    # restore model partially if not pretraining
    if not FLAGS.pretrain:
        ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.model + 'AE/')
        model.load_transfer_params(ckpt, None)

    # get the loss function and optimizers
    criterion = get_loss_fn()
    optimizer = get_optimizer(model.get_trainable_parameters())

    # start watching the model for gradient info
    wandb.watch(model)

    # run the training script
    for epoch in range(1, FLAGS.epochs + 1):

        # initialize the total epoch loss values
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # set model to train mode
        model.train()

        #  calculate training loss, update params
        count_train = 0
        for i_batch, batch in enumerate(train_dataloader):
            # zero prev gradients
            optimizer.zero_grad()

            # get train input and labels
            data, targets = get_inputs(batch)

            # forward pass through the network
            predictions = model(data)

            # calculate loss
            batch_loss = criterion(predictions, targets, model.parameters(), FLAGS.lmd)
            epoch_train_loss += batch_loss
            count_train += 1

            # calculate gradients
            batch_loss.backward()
            optimizer.step()

        # set the model to evaluation mode
        model.eval()

        # calculate validation loss
        with torch.no_grad():
            count_test = 0
            for i_batch, batch in enumerate(test_dataloader):
                # get train input and labels
                data, targets = get_inputs(batch)

                # forward pass through the network
                predictions = model(data)

                # calculate loss
                val_loss = meanJointPoseError(predictions, targets)
                epoch_val_loss += val_loss
                count_test += 1

        # log the metrics
        wandb.log({
            'train_loss': epoch_train_loss / count_train,
            'val_loss': epoch_val_loss / count_test
        })

        if epoch % FLAGS.ckpt == 0:
            if FLAGS.pretrain:
                ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/AE/')
            else:
                ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/ME/')
            model.save_model(ckpt, epoch)

    run.finish()


if __name__ == '__main__':
    app.run(main)
