#!/user/bin/env python3
import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np


input_tensor = K.Input(shape=(32, 32, 3))
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train = K.applications.vgg16.preprocess_input(x_train)
y_train = K.utils.to_categorical(y_train, 10)
x_test = K.applications.vgg16.preprocess_input(x_test)
y_test = K.utils.to_categorical(y_test, 10)

x_train = np.concatenate((x_train, np.flip(x_train, 2)), 0)
y_train = np.concatenate((y_train, y_train), 0)

model = K.applications.VGG16(include_top=False,
                             pooling='max',
                             input_tensor=input_tensor,
                             weights='imagenet')

output = model.get_layer('block3_pool').output
x = K.layers.GlobalAveragePooling2D()(output)
x = K.layers.BatchNormalization()(x)
x = K.layers.Dense(256, activation='relu')(x)
x = K.layers.Dropout(0.4)(x)
x = K.layers.Dense(128, activation='relu')(x)
x = K.layers.Dropout(0.2)(x)
output = K.layers.Dense(10, activation='softmax')(x)

model = K.models.Model(model.input, output)

lrr = K.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                    factor=.01,
                                    patience=3,
                                    min_lr=1e-5)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=128,
                    callbacks=[lrr],
                    epochs=30,
                    verbose=1)

model.save('cifar10.h5')

f, ax = plt.subplots(2, 1, figsize=(10, 15))

ax[0].plot(model.history.history['loss'], color='b', label='Training Loss')
ax[0].plot(model.history.history['val_loss'],
           color='r',
           label='Validation Loss')

ax[1].plot(model.history.history['acc'],
           color='b',
           label='Training  Accuracy')
ax[1].plot(model.history.history['val_acc'],
           color='r',
           label='Validation Accuracy')
