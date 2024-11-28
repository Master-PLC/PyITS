import logging
import os

COLOR_DICT = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
}


class Logger:
    def __init__(self, log_dir, name=None, remove_old=True):
        if name is None:
            name = __name__

        os.makedirs(log_dir, exist_ok=True) 
        filename = os.path.join(log_dir, 'log.log')
        if remove_old and os.path.exists(filename):
            os.remove(filename)

        self.filename = filename
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # remove all potential handlers in case that handlers not closed before, especially in notebook environment
        if len(self.logger.handlers) > 0:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        handler02 = logging.StreamHandler()
        handler02.setLevel(logging.DEBUG)
        handler02.setFormatter(formatter)
        self.logger.addHandler(handler02)

    def log(self, level, message, color=None):
        self.logger.log(level, wrap_message(message, color))

    def info(self, message, color=None):
        self.logger.info(wrap_message(message, color))

    def debug(self, message, color=None):
        self.logger.debug(wrap_message(message, color))

    def remove_handles(self):
        try:
            if self.logger is not None:
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
                print('Logger close successfully.')
            else:
                print('Logger is None. Do nothing.')
        except:
            print('Exception in closing logger')


def wrap_message(message, color=None):
    if color is None or color not in COLOR_DICT:
        return message
    else:
        color_prefix = COLOR_DICT[color]
        return color_prefix + message + '\033[0m'
