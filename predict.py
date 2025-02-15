import pickle

model_weights_file = 'mlp_weights.pickle'

with open(model_weights_file, 'rb') as file:
	restored_params = pickle.load(file)

from PIL import Image
import numpy as np

# Step 2: Process the uploaded image
def load_and_preprocess_image(file_path):
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))               # Resize to 28x28
    img = np.array(img) / 255.0              # Normalize pixel values to [0, 1]
    img = img.reshape(28*28*1,)          # Add batch and channel dimensions
    return img

def predict(params, image):
    """Function for per-example predictions."""
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = swish(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits

test_image = load_and_preprocess_image('5.jpg')

import jax.numpy as jnp
from jax.nn import swish

test_image_jax = jnp.array(test_image, dtype=jnp.float32)

# logits = state.apply_fn({'params': state.params}, test_image_jax)
# prediction = jnp.argmax(logits, axis=-1)
preds = predict(restored_params, test_image_jax)
print(preds.shape)
print(preds)
print(jnp.argmax(preds))

test_image = load_and_preprocess_image('1.jpg')
test_image_jax = jnp.array(test_image, dtype=jnp.float32)
preds = predict(restored_params, test_image_jax)
print(preds)
print(jnp.argmax(preds))

test_image = load_and_preprocess_image('0.jpg')
test_image_jax = jnp.array(test_image, dtype=jnp.float32)
preds = predict(restored_params, test_image_jax)
print(preds)
print(jnp.argmax(preds))

test_image = load_and_preprocess_image('2.jpg')
test_image_jax = jnp.array(test_image, dtype=jnp.float32)
preds = predict(restored_params, test_image_jax)
print(preds)
print(jnp.argmax(preds))

'''
(10,)
[ 1.4876299 -1.0146598  1.7105988  3.2156782  2.06742    4.8191853
  3.5054214  2.407446   2.0020928  4.878855 ]
9
[ -1.258486    -7.9395604   -3.83097     -2.651958    -5.405367
   0.15874016  -0.20326981 -15.8506565    3.6629796   -4.124264  ]
8
[-11.567433   -3.0353298   2.1876252   2.2612288  -8.422437   -1.7705907
  -4.4906297  -5.4630218   2.3579211  -7.0774446]
8
[ -4.4067483  -16.91391     -7.408767    -4.4889984   -9.35726
  -1.5589737   -8.983349   -12.669952     0.11709702  -3.554371  ]
8
'''