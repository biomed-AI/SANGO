import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import shutil
import time
import numpy as np
import logging
from typing import Literal, Optional, Tuple, List, Any
from argparse import Namespace
from subprocess import Popen, PIPE
import torch
import random

class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self, **kwargs):
        super(StochasticReverseComplement, self).__init__()

    def forward(self, seq_1hot):
        if self.training:
            reverse_bool = torch.rand(1) > 0.5
            # reverse_bool = reverse_bool.to(seq_1hot.device)
            if reverse_bool:
                src_seq_1hot = torch.flip(seq_1hot, dims=[-1])
                src_seq_1hot = torch.flip(src_seq_1hot, dims=[1])
            else:
                src_seq_1hot = seq_1hot
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, torch.tensor(False).to(seq_1hot.device)

class SwitchReverse(nn.Module):
    """Reverse predictions if the inputs were reverse complemented."""

    def __init__(self, **kwargs):
        super(SwitchReverse, self).__init__()

    def forward(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        xd = len(x.shape)
        if xd == 2:
            rev_axes = [1]
        elif xd == 3:
            rev_axes = [1, 2]
        else:
            raise ValueError("Cannot recognize SwitchReverse input dimensions %d." % xd)

        if reverse:
            out = torch.flip(x, dims=rev_axes)
        else:
            out = x
        return out

class StochasticShift(nn.Module):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, pad="uniform", **kwargs):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
        self.pad = pad

    def forward(self, seq_1hot):
        if self.training:
            shift_i = torch.randint(0, len(self.augment_shifts), (1,))
            shift = self.augment_shifts[shift_i]

            if torch.not_equal(shift, 0):
                sseq_1hot = shift_sequence(seq_1hot, shift)
            else:
                sseq_1hot = seq_1hot

            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({"shift_max": self.shift_max, "pad": self.pad})
        return config


def shift_sequence(seq, shift, pad_value=0.25):
    """Shift a sequence left or right by shift_amount.

    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if seq.ndimension() != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.shape

    pad = pad_value * torch.ones_like(seq[:, 0 : torch.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return torch.cat([pad, sliced_seq], dim=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return torch.cat([sliced_seq, pad], dim=1)

    if torch.greater(shift, 0):
        sseq = _shift_right(seq)
    else:
        sseq = _shift_left(seq)

    return sseq

def get_run_info(argv: List[str], args: Namespace=None, **kwargs) -> str:
    s = list()
    s.append("")
    s.append("##time: {}".format(time.asctime()))
    s.append("##cwd: {}".format(os.getcwd()))
    s.append("##cmd: {}".format(' '.join(argv)))
    if args is not None:
        s.append("##args: {}".format(args))
    for k, v in kwargs.items():
        s.append("##{}: {}".format(k, v))
    return '\n'.join(s)

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir


def make_logger(
        title: Optional[str]="", 
        filename: Optional[str]=None, 
        level: Literal["INFO", "DEBUG"]="INFO", 
        mode: Literal['w', 'a']='w',
        trace: bool=True, 
        **kwargs):
    if isinstance(level, str):
        level = getattr(logging, level)
    logger = logging.getLogger(title)
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)

    if trace is True or ("show_line" in kwargs and kwargs["show_line"] is True):
        formatter = logging.Formatter(
                '%(levelname)s(%(asctime)s) [%(filename)s:%(lineno)d]:%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(levelname)s(%(asctime)s):%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    # formatter = logging.Formatter(
    #     '%(message)s\t%(levelname)s(%(asctime)s)', datefmt='%Y%m%d %H:%M:%S'
    # )

    sh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(sh)

    if filename is not None:
        if os.path.exists(filename):
            suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(filename)))
            while os.path.exists("{}.conflict_{}".format(filename, suffix)):
                suffix = "{}_1".format(suffix)
            shutil.move(filename, "{}.conflict_{}".format(filename, suffix))
            warnings.warn("log {} exists, moved to to {}.conflict_{}.log".format(filename, filename, suffix))
        fh = logging.FileHandler(filename=filename, mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def model_summary(model):
    """
    model: pytorch model
    """
    import torch
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}
    
    
def set_seed(seed: int, force_deterministic: bool=False):
    """
    set seed
    """
    if float(torch.version.cuda) >= 10.2:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
