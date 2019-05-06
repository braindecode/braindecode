from abc import ABC, abstractmethod
import logging

log = logging.getLogger(__name__)


class Logger(ABC):
    @abstractmethod
    def log_epoch(self, epochs_df):
        raise NotImplementedError("Need to implement the log_epoch function!")


class Printer(Logger):
    """
    Prints output to the terminal using Python's logging module.
    """

    def log_epoch(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        log.info("Epoch {:d}".format(i_epoch))
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            log.info("{:25s} {:.5f}".format(key, val))
        log.info("")


class TensorboardWriter(Logger):

    """
    Logs all values for tensorboard visualiuzation using tensorboardX.
            
    Parameters
    ----------
    log_dir: string
        Directory path to log the output to
    """

    def __init__(self, log_dir):
        # import inside to prevent dependency of braindecode onto tensorboardX
        from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def log_epoch(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        last_row = epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            val = last_row[key]
            self.writer.add_scalar(key, val, i_epoch)
