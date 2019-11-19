# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DLCVDatasets import get_dataset
from HogFeatures import compute_hog
import my_models

training_size = 100
test_size = 100
x_train, y_train, x_test, y_test, class_names = get_dataset(
    dataset='mnist', training_size=training_size, test_size=test_size)

# %% Extract hog features
cell_size = (4, 4)
block_size = (2, 2)
nbins = 36
hog_train = compute_hog(cell_size, block_size, nbins, x_train)
hog_test = compute_hog(cell_size, block_size, nbins, x_test)
# Normalize training and test data
hog_maen = np.mean(hog_train)
hog_train -= hog_maen
hog_test -= hog_maen

hog_feature_length = hog_train.shape[1]

# Prepare data using tf.data
train_ds = tf.data.Dataset.from_tensor_slices(
    (hog_train, y_train)).shuffle(1000).batch(16)

test_ds = tf.data.Dataset.from_tensor_slices((hog_test, y_test)).batch(32)

# %% Create neural networks models
activation = 'relu'
my1LayerModel = my_models.My1HiddenLayerModel(activation=activation)
my2LayerModel = my_models.My2HiddenLayerModel(activation=activation)
my3LayerModel = my_models.My3HiddenLayerModel(activation=activation)
my8LayerModel = my_models.My8HiddenLayerModel(activation=activation)

# %% Choose an optimizer and loss function for training:
optimizer = tf.keras.optimizers.Adam(0.01)
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

# %% Select metrics to measure the loss and the accuracy of the model.
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# %% Use tf.GradientTape to train the models
@tf.function
def train_step_1_layer(hog_features, labels):
    with tf.GradientTape() as tape:
        predictions = my1LayerModel(hog_features)
        loss = loss_obj(labels, predictions)
    # trainable_variables = [my1LayerModel.layer_1.w, my1LayerModel.layer_1.b]
    gradients = tape.gradient(loss, my1LayerModel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my1LayerModel.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    return gradients

@tf.function
def train_step_2_layer(hog_features, labels):
    with tf.GradientTape() as tape:
        predictions = my2LayerModel(hog_features)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, my2LayerModel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my2LayerModel.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    return gradients

@tf.function
def train_step_3_layer(hog_features, labels):
    with tf.GradientTape() as tape:
        predictions = my3LayerModel(hog_features)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, my3LayerModel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my3LayerModel.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    return gradients

# @tf.function
def train_step_8_layer(hog_features, labels):
    with tf.GradientTape() as tape:
        predictions = my8LayerModel(hog_features)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, my8LayerModel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my8LayerModel.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    return gradients

# This can be shared by different models
# @tf.function
def test_step(model, hog_features, labels):
    predictions = model(hog_features)
    loss = loss_obj(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)

# %% Train the models
EPOCHS = 20

# %% Train the 1-layer model
grads_history = []
for epoch in range(EPOCHS):
    for hog_features, labels in train_ds:
        gradients = train_step_1_layer(hog_features, labels)
        
        mean_grads = [0] * 1
        for idx in range(0, 1):
            # Sum up the mean gradients of W and b at idx-th layer
            mean_grads[idx] = (gradients[idx*2].numpy().mean() + gradients[idx*2+1].numpy().mean())/2
        grads_history.append(mean_grads)
        
    for hog_features, labels in test_ds:
        test_step(my1LayerModel, hog_features, labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

plt.plot(grads_history, alpha=0.3)
plt.show()


# %% Train the 2-layer model
grads_history = []
for epoch in range(EPOCHS):
    for hog_features, labels in train_ds:
        gradients = train_step_2_layer(hog_features, labels)
        
        mean_grads = [0] * 2
        for idx in range(0, 2):
            # Sum up the mean gradients of W and b at idx-th layer
            mean_grads[idx] = (gradients[idx*2].numpy().mean() + gradients[idx*2+1].numpy().mean())/2
        grads_history.append(mean_grads)
        
    for hog_features, labels in test_ds:
        test_step(my2LayerModel, hog_features, labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

plt.plot(grads_history, alpha=0.3)
plt.show()

# %% Train the 3-layer model
grads_history = []
for epoch in range(EPOCHS):
    for hog_features, labels in train_ds:
        gradients = train_step_3_layer(hog_features, labels)
        
        mean_grads = [0] * 3
        for idx in range(0, 3):
            # Sum up the mean gradients of W and b at idx-th layer
            mean_grads[idx] = (gradients[idx*2].numpy().mean() + gradients[idx*2+1].numpy().mean())/2
        grads_history.append(mean_grads)
        
    for hog_features, labels in test_ds:
        test_step(my3LayerModel, hog_features, labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

plt.plot(grads_history, alpha =0.3)
plt.show()

# %% Train the 8-layer model
grads_history = []
for epoch in range(EPOCHS):
    for hog_features, labels in train_ds:
        gradients = train_step_8_layer(hog_features, labels)\

        mean_grads = [0] * 8
        for idx in range(0, 8):
            # Sum up the mean gradients of W and b at idx-th layer
            mean_grads[idx] = (gradients[idx*2].numpy().mean() + gradients[idx*2+1].numpy().mean())/2
        grads_history.append(mean_grads)
        
    for hog_features, labels in test_ds:
        test_step(my8LayerModel, hog_features, labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

plt.plot(grads_history, alpha=0.3)
plt.show()