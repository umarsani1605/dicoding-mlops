import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_text as tf_text

FEATURE_KEY = "review"
LABEL_KEY = "sentiment"

def transformed_name(key: str) -> str:
    """Ubah nama fitur hasil transformasi.
    
    Args:
        key (str): Nama fitur asli.
    
    Returns:
        str: Nama fitur yang telah ditransformasi.
    """
    return key + "_xf"

def preprocessing_fn(inputs: dict) -> dict:
    """Fungsi preprocessing untuk mentransformasi fitur input.
    
    Args:
        inputs (dict): Kamus fitur input.
    
    Returns:
        dict: Kamus fitur yang telah ditransformasi.
    """
    outputs = {}

    # Transformasi fitur review menjadi huruf kecil
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])

    # Transformasi label menjadi bilangan bulat (0 atau 1)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(inputs[LABEL_KEY], tf.constant("positive", dtype=tf.string)),
        tf.int64
    )

    return outputs