import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model

# My own dense layer by subclassing
class MyDenseLayer(layers.Layer):
    def __init__(self, units=32, activation='sigmoid'):
        super(MyDenseLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True)

    def call(self, inputs):
        if self.activation == 'sigmoid':
            return tf.nn.sigmoid(tf.matmul(inputs, self.w) + self.b)
        else:
            return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

# My multi-layer perceptrons by subclassing
# My multi-layer perceptron by subclassing
class My1HiddenLayerModel(Model):
    def __init__(self, activation='sigmoid'):
        super(My1HiddenLayerModel, self).__init__()
        self.layer_1 = MyDenseLayer(units=20, activation=activation)
        self.output_layer = layers.Dense(10, activation='softmax')
    
    def call(self, input):
        x = self.layer_1(input)
        return self.output_layer(x)

class My2HiddenLayerModel(Model):
    def __init__(self, activation='sigmoid'):
        super(My2HiddenLayerModel, self).__init__()
        self.layer_1 = MyDenseLayer(units=20, activation=activation)
        self.layer_2 = MyDenseLayer(units=20, activation=activation)
        self.output_layer = layers.Dense(10, activation='softmax')

    def call(self, input):
        x = self.layer_1(input)
        x = self.layer_2(x)
        return self.output_layer(x)

class My3HiddenLayerModel(Model):
    def __init__(self, activation='sigmoid'):
        super(My3HiddenLayerModel, self).__init__()
        self.layer_1 = MyDenseLayer(units=20, activation=activation)
        self.layer_2 = MyDenseLayer(units=20, activation=activation)
        self.layer_3 = MyDenseLayer(units=20, activation=activation)
        self.output_layer = layers.Dense(10, activation='softmax')

    def call(self, input):
        x = self.layer_1(input)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.output_layer(x)

class My8HiddenLayerModel(Model):
    def __init__(self, activation='sigmoid'):
        super(My8HiddenLayerModel, self).__init__()
        self.layer_1 = MyDenseLayer(units=20, activation=activation)
        self.layer_2 = MyDenseLayer(units=20, activation=activation)
        self.layer_3 = MyDenseLayer(units=20, activation=activation)
        self.layer_4 = MyDenseLayer(units=20, activation=activation)
        self.layer_5 = MyDenseLayer(units=20, activation=activation)
        self.layer_6 = MyDenseLayer(units=20, activation=activation)
        self.layer_7 = MyDenseLayer(units=20, activation=activation)
        self.layer_8 = MyDenseLayer(units=20, activation=activation)
        self.output_layer = layers.Dense(10, activation='softmax')
    
    def call(self, input):
        x = self.layer_1(input)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        return self.output_layer(x)
        
if __name__ == '__main__':
    model = My1HiddenLayerModel()
    prediction = model(np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32))
    print(model.layer_1.w.numpy())
    print(model.layer_1.b.numpy())

