import os
from dataset_RGB import TrainSetLoader, ValSetLoader,TrainSetLoader2, ValSetLoader2

def get_training_data(rgb_dir, scale_factor):
    assert os.path.exists(rgb_dir)
    return TrainSetLoader(rgb_dir, scale_factor)



def get_validation_data(rgb_dir, scale_factor):
    assert os.path.exists(rgb_dir)
    return ValSetLoader(rgb_dir, scale_factor)


def debu():
    from config import Config
    opt = Config('training.yml')
    train_dir = opt.TRAINING.TRAIN_DIR
    scale_factor = opt.TRAINING.SCALE
    train_dataset = get_training_data(train_dir, scale_factor)
    print(len(train_dataset))
    print("1")

# debu()
