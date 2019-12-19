import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras import regularizers
import os

class Cifar10Recognition:

    def __init__(self):
        print("Empty Object of Class 'Cifar10Recognition' created..!")

    def initialise_parameters(self, para):
        self.__batch_size = para['batch_size']
        self.__epochs = para['epochs']
        self.__img_rows, self.__img_cols = para['img_rows'], para['img_cols']
        self.__num_of_classes = para['num_of_classes']
        self.__samples_per_class = para['samples_per_class']
        self.__total_samples = self.__num_of_classes * self.__samples_per_class
        self.__input_shape = (self.__img_rows, self.__img_cols, 1)
        self.__weight_decay = para['weight_decay']
        self.__learning_rate = para['learning_rate']
        self.__decay = para['decay']
        self.__save_dir = para['save_dir']
        self.__model_name = para['model_name']
        self.__number_of_images = para['number_of_images']
        self.__labels = para['labels']

        print("\nParameters initialised successfully..!")
        return self

    def load_data(self):
        (self.__x_train, self.__y_train), (self.__x_test, self.__y_test) = cifar10.load_data()
        print('x_train shape:', self.__x_train.shape)
        print(self.__x_train.shape[0], 'train samples')
        print(self.__x_test.shape[0], 'test samples')
        print("\nData loaded successfully..!")

    def get_batch_size(self):
        return self.__batch_size

    def get_x_train_data(self):
        return self.__x_train[:2]

    def get_y_test_data(self):
        return self.__y_test[:2]

    def prepare_data(self):
        # Normalise the data and convert
        self.__x_train = self.__x_train.astype('float32') / 255
        self.__x_test = self.__x_test.astype('float32') / 255

        # Convert labels to one-hot vectors
        self.__y_train = tf.keras.utils.to_categorical(self.__y_train, self.__num_of_classes)  # or use tf.one_hot()
        self.__y_test = tf.keras.utils.to_categorical(self.__y_test, self.__num_of_classes)

        print("\nData Prepared successfully..!")

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.__weight_decay),
                         input_shape=self.__x_train.shape[1:]))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.__weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.__weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.__weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.__weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.__weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(self.__num_of_classes, activation='softmax'))

        print("\nModel Created successfully..!")
        return model

    def compile_model(self, model):
        # initialise the optimiser
        opt = tf.keras.optimizers.RMSprop(lr=self.__learning_rate, decay=self.__decay)

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        print("\nModel Compiled successfully..!")
        return model

    def train_model(self, model):
        print('Using data augmentation in real-time.')
        # Preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            rotation_range=10,  # randomly rotate images in the range 0 to 10 degrees
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            validation_split=0.1)

        # datagen.fit(x_train)
        # (this is only needed if any of the feature-wise normalizations i.e.
        # std, mean, and principal components ZCA whitening are set to True.)

        # set things up to halt training if the accuracy  has stopped increasing
        # could also monitor = 'val' or monitor =
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto',
                                                    baseline=None, restore_best_weights=False)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(self.__x_train, self.__y_train, batch_size=self.__batch_size),
                            epochs=self.__epochs, callbacks=[callback])

        # Save model and weights
        if not os.path.isdir(self.__save_dir):
            os.makedirs(self.__save_dir)

        model_path = os.path.join(self.__save_dir, self.__model_name)
        model.save(model_path)
        print('Model saved at: %s ' % model_path)

        return model

    def evaluate_model(self, model):
        # Evaluate our trained model.
        scores = model.evaluate(self.__x_test, self.__y_test, verbose=1)

        print('\nTest Loss:', scores[0], '   Test Accuracy:', scores[1]*100,"\n")

    def predict_class(self, model):
        number_of_images = 5
        indices = tf.argmax(input=model.predict(self.__x_test[:number_of_images * number_of_images]), axis=1).numpy()
        i = 0
        print('Learned  True')
        print('==================================')
        print("\tActual : \t Predicted")
        print('==================================')
        for index in indices:
            print("Img:",i+1, self.__labels[index], ": ", self.__labels[int(self.__y_test[i][0])])
            i += 1


print("Class updated..!")