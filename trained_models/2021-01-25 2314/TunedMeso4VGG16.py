def TunedMeso4VGG16(input_shape):

    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation = 'relu'))
    model.add(Conv2D(16, (7, 7), padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(80, (5, 5), padding='same', activation = 'relu'))
    model.add(Conv2D(96, (7, 7), padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(80, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (5, 5), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (5, 5), padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, (5, 5), padding='same', activation = 'relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(112, (3, 3), padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(224, (5, 5), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(112, (5, 5), padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(56))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(120))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))

    return model, 'TunedMeso4VGG16'


