from pickletools import optimize

from tensorflow import keras


    
def create_simple_rnn(x_train, y_train,units, drop=0, rec_drop=0):

    input_layer = keras.layers.Input(batch_shape=[1, x_train.shape[1], x_train.shape[2]])
    hidden_layer_1 = keras.layers.SimpleRNN(units=units, dropout=drop, recurrent_dropout=rec_drop, stateful=True, return_sequences=True)(input_layer)
    hidden_layer_2 = keras.layers.SimpleRNN(units=units, dropout=drop, recurrent_dropout=rec_drop, stateful=True)(hidden_layer_1)
    
    output_layer = keras.layers.Dense(units=y_train.shape[1])(hidden_layer_2)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    return model


def create_complex_rnn(x_train, y_train,units, drop=0, rec_drop=0):
    input_layer = keras.layers.Input(batch_shape=[1, x_train.shape[1], x_train.shape[2]])
    lstm_1 = keras.layers.LSTM(units=units, dropout=drop, recurrent_dropout=rec_drop, stateful=True, return_sequences=True)(input_layer)
    lstm_2 = keras.layers.LSTM(units=units, dropout=drop, recurrent_dropout=rec_drop, stateful=True)(lstm_1)
    
    output_layer = keras.layers.Dense(activation='softmax',units=y_train.shape[1])(lstm_2)
    model=keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    return model




def train_rnn(model, x_train, y_train, nb_epoch):
    
    model.fit(x_train, y_train, batch_size=1, epochs=nb_epoch, verbose=0)
