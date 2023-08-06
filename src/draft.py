

import argparse

from utils import *
from Fed import *
from main import update_args_with_dict
from os import listdir
from os.path import isdir, join
import torchvision
import torchvision.transforms as transforms 

from keras.models import load_model
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer


transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)


exp_dir = RESULTS_PATH
experiments = [join(exp_dir, f) for f in listdir(exp_dir) if isdir(join(exp_dir, f))]
for exp in experiments : 
    model_path = join(exp, 'model.h5')

    if not os.path.exists(model_path):
        continue

    # load model
    model = load_model(model_path, custom_objects={'DPKerasSGDOptimizer': DPKerasSGDOptimizer})
    n_batches = 100 
    all_true, all_pred = [], []
    for batch in np.split(testset, n_batches) :
        x_test, y_test = batch
        y_test = np.argmax(y_test, axis=1)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        all_true.append(y_test)
        all_pred.append(y_pred)
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    acc = np.mean(all_true == all_pred)
    print("acc:", acc)


    print(exp) 

############################################# 
