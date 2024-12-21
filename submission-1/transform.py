import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_text as tf_text

FEATURE_KEY = "review"
LABEL_KEY = "sentiment"

def transformed_name(key):
    return key + "_xf"

def preprocessing_fn(inputs):
    outputs = {}

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(inputs[LABEL_KEY], tf.constant("positive", dtype=tf.string)),
        tf.int64
    )

    return outputs