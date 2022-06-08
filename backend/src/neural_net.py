#!/usr/bin/env python3
# coding: utf-8

import os
import time
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras import optimizers 
from keras import regularizers
from keras import initializers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, BatchNormalization, Activation, Softmax
from keras.callbacks import TensorBoard, CSVLogger
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.optimizer_v2.adadelta import Adadelta as Adadelta


def get_opt(name, lr):
    if name == 'sgd':
        opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, 
                             nesterov=True)
    elif name == 'adam':
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                              epsilon=None, decay=1e-6, amsgrad=False)
    elif name == 'adamax':
        opt = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0)
    elif name == 'adagrad':
        opt = optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
    elif name == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    return opt


def deep_nn(X, y, label, path=None):

    K.clear_session()

    try:
        path += ('/' + label)
        os.mkdir(path)
    except FileExistsError:
        print(path, 'already exists.')

    # Globals
    lr = 0.0001
    epochs = 50
    batch_size = 50
    OPT = 'adadelta'

    t = time.time()
    dt = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
    name = '_'.join([OPT, str(epochs), str(batch_size), dt])

    # Calculate class weights to improve accuracy 
    class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
    swm = np.array([class_weights[i] for i in y])

    # Convert target to categorical
    y = to_categorical(y, num_classes=len(class_weights))

    # Размер входа и выхода равен размерности данных
    in_size = X.shape[1]

    # Размеры скрытых слоев
    hidden_1_size = 400
    hidden_2_size = 150
    hidden_3_size = 50

    # Modify this when increasing artist list target
    out_size = y.shape[1]

    # Разделение на train/test/validation
    print('Splitting to train, test, and validation sets...')
    X_train, X_test, X_valid = np.split(X, [int(.6 * len(X)), int(.8 * len(X))])
    y_train, y_test, y_valid = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])

    model = Sequential()

    def add_hidden_layer(s, b=False, a=0.3, d=0.0):
        if d > 0:
            model.add(Dense(s, kernel_initializer='normal',
                            kernel_constraint=maxnorm(3)))
        else:
            model.add(Dense(s))
        if b:
            model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=a))
        # model.add(Dropout(d))

    # Входной слой
    model.add(Dense(in_size, input_shape=(in_size,), activation='relu',
                    kernel_initializer='normal', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # Скрытые слои
    add_hidden_layer(in_size, False, 0.3, 0.5)
    add_hidden_layer(hidden_1_size, True, 0.3, 0.3)
    add_hidden_layer(hidden_2_size, True, 0.3, 0.3)
    add_hidden_layer(hidden_3_size, False, 0.3, 0.1)

    # Выходной слой
    model.add(Dense(out_size))
    model.add(BatchNormalization())
    model.add(Softmax(axis=-1))


    opt = get_opt(OPT, lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  # metrics=['accuracy','msle'],
                  metrics=['accuracy'],
                  sample_weight_mode=swm)

    tensorboard = TensorBoard(log_dir=str('./logs/'+label+'/'+name+'.json'),
                              histogram_freq=1,
                              write_graph=True,
                              write_images=False)   

    csv_logs = CSVLogger(filename=path+'/logs.csv',
                         separator=',',
                         append=False)      

    print('Training...') 
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
              epochs=epochs, batch_size=batch_size, verbose=1, 
              shuffle=True, callbacks=[csv_logs])

    print('Evaluating...')
    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score)

    print('Saving model...')
    # Save model structure json
    model_json = model.to_json()
    with open(path + '/model.json', 'w') as file:
        file.write(model_json)
    # Save weights as h5
    model.save_weights(path + '/weights.h5')
    # Save sample weight mode
    np.savetxt(path + '/sample_weights.csv', swm, delimiter=',')
    # Save hyperparams
    with open(path + '/hyperparams.csv', 'w') as file:
        file.write(','.join([str(lr), OPT]))
    print('Model saved to disk')

    return model


# Загрузить модель из файлов
def load_model(path):
    # Архитектура
    with open(path + '/model.json','r') as file:
        structure = file.read()
    model = model_from_json(structure)

    # Веса
    model.load_weights(path + '/weights.h5')

    # Get sample weights for compiler
    swm = np.genfromtxt(path + '/sample_weights.csv', delimiter=',')

    # Гиперпараметры
    with open(path + '/hyperparams.csv', 'r') as file:
        LR, OPTIMIZER_NAME = file.read().split(',')
        
    opt = get_opt(OPTIMIZER_NAME, float(LR))

    # Compile the loaded model
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],
              # sample_weight_mode=swm
              )

    return model





