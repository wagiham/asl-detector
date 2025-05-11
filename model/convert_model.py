# import tensorflow as tf
# import os
#
# # Correct absolute path to the model
# model_path = os.path.abspath("Model/keras_model.h5")  # FIXED
# fixed_model_path = os.path.abspath("Model/keras_model_fixed")
#
# print("Loading model from:", model_path)
#
# # Load the model
# model = tf.keras.models.load_model(model_path)
#
# # Save the model in a new format
# model.save(fixed_model_path)
#
# print("Model successfully converted and saved at:", fixed_model_path)
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model_path = "model/keras_model.h5"
model = load_model(model_path, compile=False)

# Fix DepthwiseConv2D layers
for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        config = layer.get_config()
        if "groups" in config:
            del config["groups"]  # Remove the unsupported parameter
        model.layers[i] = tf.keras.layers.DepthwiseConv2D.from_config(config)

# Save the fixed model to the same file (overwrite original)
model.save(model_path)

print("✅ Model has been fixed and saved as 'keras_model.h5'")
