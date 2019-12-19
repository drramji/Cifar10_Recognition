import tensorflow as tf
from cifar10_recognition.cifar_recognition import Cifar10Recognition
from cifar10_recognition.init_parameters import get_init_parameters
import os

def evaluate_model(self):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_trained_model.h5'
    model_path = os.path.join(save_dir, model_name)
    model = tf.keras.models.load_model(model_path)
    print(model.summary)

    cifar_obj = Cifar10Recognition()

    init_parameters = get_init_parameters()
    cifar_obj.initialise_parameters(init_parameters)
    cifar_obj.load_data()
    cifar_obj.prepare_data()
    cifar_obj.evaluate_model(model)