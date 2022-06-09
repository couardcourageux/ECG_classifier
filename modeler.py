import tensorflow as tf

def display_model(model):
    # affiche le résumé du modèle
    model.summary()
    # en créé une image
    return tf.keras.utils.plot_model(model, show_shapes=True)



def compile_model(model, mode='lstm', learning_r=0.01):
    """function to compile with preset parameters
    """
    
    if mode == 'lstm':
        optimizer_alg = tf.keras.optimizers.Adam(learning_rate=learning_r)
        model.compile(optimizer=optimizer_alg, loss='mean_squared_error', metrics=["accuracy" ,"loss"])

    if mode == 'cnn':
        optimizer_alg = tf.keras.optimizers.SGD(learning_rate=learning_r)
        model.compile(optimizer=optimizer_alg, loss='categorical_crossentropy', metrics=["accuracy", "loss"])