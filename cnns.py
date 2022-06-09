from tensorflow import keras





def create_cnn(x_train, y_train, nb_layers=1, drop=0):
    # permet de conserver la forme des données au fil de leur passage dans le cnn
    padding = 'same'
    
    # valeur du décalage appliqué à chaque filtre
    stride = 1
    
    # taille du noyau de convolution utilisé par les filtres
    kernel_size = 15
    
    # nombre de filtres utilisés
    filters = 5
    
    # fonction d'activation    
    activation = 'relu'
    
    # création de la couche d'entrée, dont la forme est celle d'1 série temporelle ESG
    input_layer = keras.layers.Input(x_train.shape[1:])
    
    # première couche de convolution, appliquant les paramétres
    conv_1 = keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        activation=activation
    )(input_layer)
    
    # couche de réduction du résultat de la convolution
    pooling_1 = keras.layers.MaxPooling1D(pool_size = 2, strides = 2, 
                                          padding='valid')(conv_1)
    
    # si on veut un modèle 2 couches, on réitère le processus
    if nb_layers == 2 :
    
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
        
    else:
        flattened = keras.layers.Flatten()(pooling_1)
        
    # flattened est une couche transformant la forme de la sortie, la renverse pour la rendre 
    # compatible    
        
    # couche de sortie, units = nb de classes, ici y_train.shape[1]
    output_layer = keras.layers.Dense(units=y_train.shape[1], activation='softmax')(flattened)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    # le modèle est renvoyé
    return model
    