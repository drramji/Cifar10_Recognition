from cifar10_recognition.cifar_recognition import Cifar10Recognition
from cifar10_recognition.init_parameters import get_init_parameters


def train_cnn():
    cifar_obj = Cifar10Recognition()

    init_parameters = get_init_parameters()
    cifar_obj.initialise_parameters(init_parameters)
    cifar_obj.load_data()
    cifar_obj.prepare_data()

    model = cifar_obj.create_model()
    model = cifar_obj.compile_model(model)
    # self.train_model(model)

    print("\nModel got trained..!")