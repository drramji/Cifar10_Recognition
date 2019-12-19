import os
from cifar10_recognition.cifar_recognition import Cifar10Recognition

def get_init_parameters():
    ## Parameter setting
    parameter_dict = {
        'batch_size' : 32,
        'img_rows' : 32,
        'img_cols' : 32,
        'num_of_classes' : 10, #len(filenames),
        'samples_per_class' : 1000,
        'epochs' : 2,
        'weight_decay' : 1e-4,
        'learning_rate' : 0.002,
        'decay' : 1e-6,
        'save_dir' : os.path.join(os.getcwd(), 'saved_models'),
        'model_name' : 'cifar10_trained_model.h5',
        'number_of_images' : 5,
        'labels' : ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    }
    return parameter_dict
    # cifar_obj.initialise_parameters(parameter_dict)