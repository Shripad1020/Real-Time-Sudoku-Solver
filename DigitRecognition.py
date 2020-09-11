import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


class DigitRecognition :
    # load train and test dataset
    def load_dataset():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # normalizing data
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)
        
        return x_train, y_train, x_test, y_test
    
    # define cnn model
    def cnn_model():
        model = Sequential()
        model.add(Flatten(input_shape = (28,28)))
        model.add(Dense(units = 128, activation = tf.nn.relu))
        model.add(Dense(units = 128, activation = tf.nn.relu))  
        model.add(Dense(units = 10, activation = tf.nn.softmax))
        
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        
        return model


def main():
    x_train, y_train, x_test, y_test = DigitRecognition.load_dataset()
    model = DigitRecognition.cnn_model()

    # fit model
    model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 5)
    
    # Final evaluation of the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy =  {}".format(accuracy))
    print("Loss =  {}".format(loss))
    
    # save model
    model.save("DigitRecognition.model")
    
    
if __name__ == '__main__':
    main()