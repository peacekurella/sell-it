import os
import wandb
import torch
from torch.utils.data import DataLoader

from absl import app
from absl import flags

from HagglingDataset import HagglingDataset
from BodyAE import BodyAE
from BodyMotionGenerator import BodyMotionGenerator
from losses import reconstruction_l1

FLAGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('train', 'Data/train/', 'Directory containing train files')
flags.DEFINE_string('test', 'Data/test/', 'Directory containing train files')
flags.DEFINE_string('ckpt_dir', 'ckpt/', 'Directory to store checkpoints')

flags.DEFINE_integer('batch_size', 64, 'Training set mini batch size')
flags.DEFINE_integer('epochs', 20, 'Training epochs')
flags.DEFINE_integer('ckpt', 2, 'Number of epochs to checkpoint')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('lmd', 0.001, 'L1 Regularization factor')

flags.DEFINE_bool('pretrain', True, 'pretrain the auto encoder')
flags.DEFINE_bool('resume_train', False, 'Resume training the model')


def get_inputs(batch):
    """
    Generates the inputs to the netwo
    :param batch: batch for training
    :return: train_x, train_y tensors
    """
    b = batch['buyer']['joints21']
    l = batch['leftSeller']['joints21']
    r = batch['rightSeller']['joints21']

    if FLAGS.pretrain:
        train_x = torch.cat((b, l, r), dim=0).permute(0, 2, 1).float().cuda()
        train_y = torch.cat((b, l, r), dim=0).permute(0, 2, 1).float().cuda()
        return train_x, train_y
    else:
        set_x_a = torch.cat((b, l), dim=2)
        set_x_b = torch.cat((b, r), dim=2)
        train_x = torch.cat((set_x_a, set_x_b), dim=0).permute(0, 2, 1).float().cuda()
        train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float().cuda()
        return train_x, train_y


def get_model():
    """
    Returns the appropriate model for training
    :return: PyTorch model that extends nn.Module
    """

    if FLAGS.pretrain:
        return BodyAE(FLAGS).cuda()
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

    return torch.optim.Adamax(parameters, lr=FLAGS.learning_rate)


def get_scheduler(optimizer):
    """
    Returns the LR scheduler
    :param optimizer: Pytorch optimizer
    :return: pytorch lr scheduler
    """

    return torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)


def main(args):
    # initialize the dataset and the data loader
    train_dataset = HagglingDataset(FLAGS.train, FLAGS)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=10)
    test_dataset = HagglingDataset(FLAGS.test, FLAGS)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=10)

    # initialize the model
    model = get_model()
    run = wandb.init(project='bodyAE', reinit=True)

    # restore model if needed
    if FLAGS.resume_train:
        if FLAGS.pretrain:
            ckpt = os.path.join(FLAGS.ckpt_dir, 'AE/')
        else:
            ckpt = os.path.join(FLAGS.ckpt_dir, 'ME/')
        model.load_model(ckpt, None)

    # restore model partially if not pretraining
    if not FLAGS.pretrain:
        ckpt = os.path.join(FLAGS.ckpt_dir, 'AE/')
        model.load_transfer_params(ckpt, None)

    # get the loss function and optimizers
    criterion = get_loss_fn()
    optimizer = get_optimizer(model.get_trainable_parameters())

    # start watching the model for gradient info
    wandb.watch(model)

    # run the training script
    for epoch in range(1, FLAGS.epochs+1):

        # initialize the total epoch loss values
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        #  calculate training loss, update params
        count_train = 0
        for i_batch, batch in enumerate(train_dataloader):
            # set model to train mode
            model.train()
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

        # calculate validation loss
        count_test = 0
        for i_batch, batch in enumerate(test_dataloader):
            # set the model to evaluation mode
            model.eval()

            # get train input and labels
            data, targets = get_inputs(batch)

            # forward pass through the network
            predictions = model(data)

            # calculate loss
            val_loss = criterion(predictions, targets, model.parameters(), FLAGS.lmd)
            epoch_val_loss += val_loss
            count_test += 1

        # log the metrics
        wandb.log({
            'train_loss': epoch_train_loss / count_train,
            'val_loss': epoch_val_loss / count_test
        })

        if epoch % FLAGS.ckpt == 0:
            if FLAGS.pretrain:
                ckpt = os.path.join(FLAGS.ckpt_dir, 'AE/')
            else:
                ckpt = os.path.join(FLAGS.ckpt_dir, 'ME/')
            model.save_model(ckpt, epoch)

    run.finish()

if __name__ == '__main__':
    app.run(main)
