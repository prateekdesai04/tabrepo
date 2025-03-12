"""
**logger** module just exposes a ``setup`` function to quickly configure the python logger.
"""
import datetime as dt
import io
import logging
import sys

# prevent asap other modules from defining the root logger using basicConfig
logging.basicConfig(handlers=[logging.NullHandler()])

utils_logger = logging.getLogger('tabrepo.utils.experiment_utils_v6')
scripts_logger = logging.getLogger('tabrepo.scripts_v6.abstract_class')
singular_logger = logging.getLogger('singular_model')

logging.TRACE = logging.TRACE if hasattr(logging, 'TRACE') else 5


class MillisFormatter(logging.Formatter):
    converter = dt.datetime.fromtimestamp  # type: ignore

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            t = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
        s = "%s.%03d" % (t, record.msecs)
        return s


def setup(log_file=None, root_file=None, root_level=logging.WARNING, app_level=None, console_level=None,
          print_to_log=False):
    """
    configures the Python logger.
    :param log_file:
    :param root_file:
    :param root_level:
    :param app_level:
    :param console_level:
    :return:
    """
    logging.captureWarnings(True)
    # warnings = logging.getLogger('py.warnings')

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    root = logging.getLogger()
    root.setLevel(root_level)

    app_level = app_level if app_level else root_level
    console_level = console_level if console_level else app_level

    # create console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    utils_logger.addHandler(console)
    utils_logger.setLevel(app_level)
    scripts_logger.addHandler(console)
    scripts_logger.setLevel(app_level)

    file_formatter = MillisFormatter('[%(levelname)s] [%(name)s:%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    if log_file:
        # create file handler
        app_handler = logging.FileHandler(log_file, mode='a')
        app_handler.setLevel(app_level)
        app_handler.setFormatter(file_formatter)
        utils_logger.addHandler(app_handler)
        scripts_logger.addHandler(app_handler)

    if root_file:
        root_handler = logging.FileHandler(root_file, mode='a')
        root_handler.setLevel(root_level)
        root_handler.setFormatter(file_formatter)
        root.addHandler(root_handler)

    if print_to_log:
        import builtins
        nl = '\n'
        print_logger = logging.getLogger(utils_logger.name + '.print')
        buffer = dict(out=None, err=None)

        ori_print = builtins.print

        def new_print(*args, sep=' ', end=nl, file=None):
            if file not in [None, sys.stdout, sys.stderr]:
                return ori_print(*args, sep=sep, end=end, file=file)

            nonlocal buffer
            buf_type = 'err' if file is sys.stderr else 'out'
            buf = buffer[buf_type]
            if buf is None:
                buf = buffer[buf_type] = io.StringIO()
            line = sep.join(map(str, [*args]))
            buf.write(line)  # "end" newline always added by logger
            if end == nl or line.endswith(nl):  # flush buffer for every line
                with buf:
                    level = logging.ERROR if buf_type == 'err' else logging.INFO
                    print_logger.log(level, buf.getvalue())
                    buffer[buf_type] = None

        builtins.print = new_print


def singular_setup(individual_log_file=None, individual_level=logging.DEBUG, print_to_log=True):
    ind_console = logging.StreamHandler()
    ind_console.setLevel(individual_level)
    singular_logger.addHandler(ind_console)

    file_formatter = MillisFormatter('[%(levelname)s] [%(name)s:%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    individual_handler = logging.FileHandler(individual_log_file, mode='a')
    individual_handler.setLevel(individual_level)
    individual_handler.setFormatter(file_formatter)
    singular_logger.addHandler(individual_handler)

    if print_to_log:
        import builtins
        nl = '\n'
        print_logger = logging.getLogger(utils_logger.name + '.print')
        singular_print_logger = logging.getLogger(singular_logger.name + '.print')
        buffer = dict(out=None, err=None)

        ori_print = builtins.print

        def new_print(*args, sep=' ', end=nl, file=None):
            if file not in [None, sys.stdout, sys.stderr]:
                return ori_print(*args, sep=sep, end=end, file=file)

            nonlocal buffer
            buf_type = 'err' if file is sys.stderr else 'out'
            buf = buffer[buf_type]
            if buf is None:
                buf = buffer[buf_type] = io.StringIO()
            line = sep.join(map(str, [*args]))
            buf.write(line)  # "end" newline always added by logger
            if end == nl or line.endswith(nl):  # flush buffer for every line
                with buf:
                    level = logging.ERROR if buf_type == 'err' else logging.INFO
                    print_logger.log(level, buf.getvalue())
                    singular_print_logger.log(level, buf.getvalue())
                    buffer[buf_type] = None

        builtins.print = new_print
