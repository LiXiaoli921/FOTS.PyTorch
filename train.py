import argparse
import json
import logging
import math
import os
import pathlib

from torch.utils.data import random_split

from data_loader import SynthTextDataLoaderFactory,dataset

# from data_loaders import collate_fn
import torch.utils.data as torchdata
from data_loader import datautils
from logger import Logger
from model.loss import *
from model.model import *
from model.metric import *
from trainer import Trainer
from utils.bbox import Toolbox


logging.basicConfig(level=logging.DEBUG, format='')


ICDAR2015_DATA_ROOT = pathlib.Path('/media/ai/64AE7354AE731DAC/dataset/ICDAR2015/')


def train_val_split(dataset, ratio: str='8:2'):
    '''

    :param ratio: train v.s. val etc. 8:2
    :param dataset:
    :return:
    '''

    try:
        train_part, val_part = ratio.split(':')
        train_part, val_part = int(train_part), int(val_part)
    except:
        print('ratio is illegal.')
        train_part, val_part = 8, 2

    train_len =  math.floor(len(dataset) * (train_part / (train_part + val_part)))
    val_len = len(dataset) - train_len

    train, val = random_split(dataset, [train_len, val_len])
    return train, val


def main(config, resume):
    train_logger = Logger()

    # Synth800K
    # data_loader = SynthTextDataLoaderFactory(config)
    # train = data_loader.train()
    # val = data_loader.val()

    # icdar 2015
    custom_dataset = dataset.MyDataset(ICDAR2015_DATA_ROOT / 'ch4_training_images',
                                        ICDAR2015_DATA_ROOT / 'ch4_training_localization_transcription_gt')

    train_dataset, val_dataset = train_val_split(custom_dataset)
    # data_loader = torchdata.DataLoader(train_dataset, collate_fn = SynthTextDataLoaderFactory.collate_fn, batch_size = 1, shuffle = True)
    data_loader = torchdata.DataLoader(train_dataset, collate_fn = datautils.collate_fn, batch_size = 8, shuffle = True)
    valid_data_loader = torchdata.DataLoader(val_dataset, collate_fn = datautils.collate_fn, batch_size = 8, shuffle = True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    model = eval(config['arch'])(config['model'])
    model.summary()

    loss = eval(config['loss'])(config['model'])
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger,
                      toolbox = Toolbox)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = 'config.json'
    with open("config.json", 'r') as load_f:
        config = json.load(load_f)
        print(config)
        print(config['gpus'])
    # config1 = json.loads(config)
    # print(config1)
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.resume)
