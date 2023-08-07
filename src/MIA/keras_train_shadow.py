# from shadow.trainer import train
from keras_make_data import make_member_nonmember, keras_make_member_nonmember
from utils.seed import seed_everything
from utils.load_config import load_config
import os
from os.path import join
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from easydict import EasyDict
import yaml
# import wandb
import importlib
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint

from keras.utils import to_categorical


# load config
config_path = 'keras_config.yaml'
CFG = load_config("CFG", config_path = config_path)
CFG_ATTACK = load_config("CFG_ATTACK", config_path = config_path)

from keras_utils import * 

# seed for future replication
seed_everything(CFG.seed)

# Load the CIFAR dataset
# CIFAR train is used for SHADOW MODEL train & evaluation whereas CIFAR test is used for TARGET MODEL train & evaluation
if CFG.dataset_name.lower() == "cifar10":
    DSET_CLASS = torchvision.datasets.CIFAR10
    CFG.num_classes = 10
elif CFG.dataset_name.lower() == "cifar100":
    DSET_CLASS = torchvision.datasets.CIFAR100
    CFG.num_classes = 100

transform = transforms.Compose(
    [
        transforms.Resize((CFG.input_resolution, CFG.input_resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


if not os.path.exists(CFG_ATTACK.shadow_models_path):
    os.makedirs(CFG_ATTACK.shadow_models_path)

for i in range(CFG.num_shadow_models, CFG.num_shadow_models * 10) : 
    trainset = DSET_CLASS(root="./data", train=True, download=True, transform=transform)
    testset = DSET_CLASS(root="./data", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset,  shuffle=True, num_workers=0)
    testloader = DataLoader(testset,  shuffle=False, num_workers=0)
    
    # subst of trainset with same length as testset
    random_indc = np.random.choice(len(trainset), len(testset), replace=False)
    small_trainset = Subset(trainset, random_indc.tolist())
    small_trainloader = DataLoader(small_trainset, shuffle=True, num_workers=0)
    #  

    train_x, train_y = next(iter(trainloader))
    test_x, test_y = next(iter(testloader))
    train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y, dtype = np.float32), np.array(test_x), np.array(test_y, dtype = np.float32)

    # change data format to channel last
    train_x = np.transpose(train_x, (0, 2, 3, 1))
    test_x = np.transpose(test_x, (0, 2, 3, 1))

    train_y = to_categorical(train_y, num_classes = CFG.num_classes)
    test_y = to_categorical(test_y, num_classes = CFG.num_classes)

    this_shadow_model_path = join(CFG_ATTACK.shadow_models_path, f"shadow_model_{i}")
    if not os.path.exists(this_shadow_model_path):
        os.makedirs(this_shadow_model_path)
    

    input_shape = train_x.shape[1:]
    model = get_cnn_keras_model(input_shape, num_classes = CFG.num_classes, compile_model = True)

    log_path = join(this_shadow_model_path, 'training.log')
    model_path = join(this_shadow_model_path, 'best_model.h5')
    attack_dset_path = join(this_shadow_model_path, f'attack_dset_shadow_{i}.csv')

    callbacks = [] 
    # callbacks.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True))
    # callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min'))
    callbacks.append(CSVLogger(log_path))
    callbacks.append(ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'))

    model.fit(x = train_x, y = train_y, batch_size = 100, epochs=CFG_ATTACK.train_epoch, validation_data=(test_x, test_y), callbacks=callbacks, verbose = 1)

    member_dset, nonmember_dset = keras_make_member_nonmember(model, small_trainloader, testloader)


    df_member = pd.DataFrame(member_dset, columns=[str(i) for i in range(CFG.num_classes)])
    df_member["is_member"] = 1
    df_non_member = pd.DataFrame(nonmember_dset, columns=[str(i) for i in range(CFG.num_classes)])
    df_non_member["is_member"] = 0

    df_attack_dset = pd.DataFrame({}, columns=[str(i) for i in range(CFG.num_classes)] + ["is_member"])
    df_attack_dset = pd.concat([df_attack_dset, df_member, df_non_member])
    df_attack_dset.to_csv(
        attack_dset_path,
        index=False,
    )
