from cifar10_recognition.cifar_recognition import Cifar10Recognition
from cifar10_recognition.cifar_evaluate import evaluate_model
from cifar10_recognition.cifar_test import test_cnn_model
from cifar10_recognition.train_cnn import train_cnn

while True or ch == "":
    print("\n# Select Choice:")
    print("# 1. Train")
    print("# 2. Test")
    print("# 3. Evaluate")
    print("# 4. Exit")
    ch = input("Enter Choice:")

    if ch != "":
        ch = int(ch)
    else:
        continue

    if ch == 1:
        print('\nTraining CNN Model..!')
        train_cnn()

    elif ch == 2:
        cifar_obj = Cifar10Recognition()
        test_cnn_model(cifar_obj)
        print('\nTesting Model ..!')

    elif ch == 3:
        print('\nModel Evaluation..!')
        cifar_obj = Cifar10Recognition()
        evaluate_model(cifar_obj)

    else:
        print('\nExited.. You completed execution..!\n')

    if(ch > 3):
        break