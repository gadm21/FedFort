

import tensorflow as tf 




def get_cnn_keras_model(input_shape, num_classes, weight_decay=0.0000, compile_model = True):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            16,
            3,
            strides=1,
            padding='same',
            activation='relu',
            input_shape= input_shape),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(
            32, 3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    if compile_model:
        model.compile(
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=['accuracy']
        )
    
    return model 



def get_model_stats(model, data, labels, loss_fn) : 
    """a function that takes a model, data, and labels and returns the predictions and losses for the data

    Args:
        model (keras model): A keras model
        data (np.array): data samples with the distributions (samples, input_shape)
        labels (np.array): labels for the data samples (samples, num_classes)
        loss_fn (keras loss function): loss function to be used for calculating the loss
    """

    predictions = model.predict(data)
    loss = loss_fn(labels, predictions)

    return predictions, loss
