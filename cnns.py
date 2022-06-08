from tensorflow import keras





def create_cnn(x_train, y_train, drop=0):
    padding = 'same'
    stride = 1
    kernel_size = 15
    filters = 6    
    activation = 'relu'
    
    input_layer = keras.layers.Input(x_train.shape[1:])
    conv_1 = keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        activation=activation
    )(input_layer)
    pooling_1 = keras.layers.MaxPooling1D(pool_size = 2, strides = 2, 
                                          padding='valid')(conv_1)
    
    conv_2 = keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        activation=activation
    )(pooling_1)
    pooling_2 = keras.layers.MaxPooling1D(pool_size = 2, strides = 2, 
                                          padding='valid')(conv_2)
    
    
    flattened = keras.layers.Flatten()(pooling_2)
    output_layer = keras.layers.Dense(units=y_train.shape[1], activation='softmax')(flattened)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    return model
    