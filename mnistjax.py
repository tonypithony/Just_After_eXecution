import tensorflow as tf
import tensorflow_datasets as tfds

data_dir = '/tmp/tfds'

data, info = tfds.load(name="mnist",
						data_dir=data_dir,
						as_supervised=True,
						with_info=True)

data_train, data_test = data['train'], data['test']


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

ROWS = 3
COLS = 10

i = 0
fig, ax = plt.subplots(ROWS, COLS)
for image, label in data_train.take(ROWS*COLS):
	ax[int(i/COLS), i%COLS].axis('off')
	ax[int(i/COLS), i%COLS].set_title(str(label.numpy()))
	ax[int(i/COLS), i%COLS].imshow(np.reshape(image, (28,28)), cmap='gray')
	i += 1
plt.savefig('somedigits.png')
plt.close()

HEIGHT = 28
WIDTH = 28
CHANNELS = 1

NUM_PIXELS = HEIGHT * WIDTH * CHANNELS
NUM_LABELS = info.features['label'].num_classes

def preprocess(img, label):
	"""Resize and preprocess images."""
	return (tf.cast(img, tf.float32)/255.0), label

train_data = tfds.as_numpy(data_train.map(preprocess).batch(32).prefetch(1))
test_data = tfds.as_numpy(data_test.map(preprocess).batch(32).prefetch(1))


from jax import random

LAYER_SIZES = [28*28, 512, 10]
PARAM_SCALE = 0.01

def init_network_params(sizes, key=random.PRNGKey(0), scale=1e-2):
	"""Initialize all layers for a fully-connected
	neural network with given sizes"""
	def random_layer_params(m, n, key, scale=1e-2):
		"""A helper function to randomly initialize
		weights and biases of a dense layer"""
		w_key, b_key = random.split(key)
		return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

	keys = random.split(key, len(sizes))
	return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

params = init_network_params(LAYER_SIZES, random.PRNGKey(0), scale=PARAM_SCALE)


import jax.numpy as jnp
from jax.nn import swish

def predict(params, image):
	"""Function for per-example predictions."""
	activations = image
	for w, b in params[:-1]:
		outputs = jnp.dot(w, activations) + b
		activations = swish(outputs)
	
	final_w, final_b = params[-1]
	logits = jnp.dot(final_w, activations) + final_b
	return logits

random_flattened_image = random.normal(random.PRNGKey(1), (28*28*1,))
preds = predict(params, random_flattened_image)
print(preds.shape)
print(preds)

random_flattened_images = random.normal(random.PRNGKey(1), (32, 28*28*1))
try:
	preds = predict(params, random_flattened_images)
except TypeError as e:
	print(e)


from jax import vmap

batched_predict = vmap(predict, in_axes=(None, 0))
batched_preds = batched_predict(params, random_flattened_images)
# print(batched_preds.shape, '\n')

'''
(10,)
dot_general requires contracting dimensions to have the same shape, got (784,) and (32,).
(32, 10)
'''

from jax.nn import logsumexp

def loss(params, images, targets):
	"""Categorical cross entropy loss function."""
	logits = batched_predict(params, images)
	log_preds = logits - logsumexp(logits)
	return -jnp.mean(targets*log_preds)


from jax import grad

INIT_LR = 1.0
DECAY_RATE = 0.95
DECAY_STEPS = 5

def update(params, x, y, epoch_number):
	grads = grad(loss)(params, x, y)
	lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
	return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]


from jax import value_and_grad

def update(params, x, y, epoch_number):
	loss_value, grads = value_and_grad(loss)(params, x, y)
	lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
	return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)], loss_value


from jax.nn import one_hot

num_epochs = 600

delta=0.001
EPOCH_STOP = 6
delta_epochs = 0
test_acc_after = 0

def batch_accuracy(params, images, targets):
	images = jnp.reshape(images, (len(images), NUM_PIXELS))
	predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
	return jnp.mean(predicted_class == targets)

def accuracy(params, data):
	accs = []
	for images, targets in data:
		accs.append(batch_accuracy(params, images, targets))
	return jnp.mean(jnp.array(accs))

import time

global_losses = []
global_train_acc = []
global_test_acc = []
for epoch in range(num_epochs):
	start_time = time.time()
	losses = []

	for x, y in train_data:
		x = jnp.reshape(x, (len(x), NUM_PIXELS))
		y = one_hot(y, NUM_LABELS)
		params, loss_value = update(params, x, y, epoch)
		losses.append(loss_value)
	epoch_time = time.time() - start_time
	
	start_time = time.time()
	
	train_acc = accuracy(params, train_data)
	test_acc = accuracy(params, test_data)
	eval_time = time.time() - start_time
	
	print(f"Epoch {epoch}/{num_epochs} in {epoch_time:.2f} sec")
	print("Eval in {:0.2f} sec".format(eval_time))
	print("Training set loss {}".format(jnp.mean(jnp.array(losses))))
	print("Training set accuracy {}".format(train_acc))
	print("Test set accuracy {}\n".format(test_acc))

	global_losses.append(jnp.mean(jnp.array(losses)))
	global_train_acc.append(train_acc)
	global_test_acc.append(test_acc)

	if (test_acc - test_acc_after) < delta:
		delta_epochs += 1
		test_acc_after = test_acc
		print(f'delta_epochs = {delta_epochs}\n\n')
	else:
		delta_epochs = 0
		test_acc_after = test_acc
	if delta_epochs == EPOCH_STOP:
		print(f"early stopping, acc_test = {test_acc:.2f}\n\n")
		break

# Plot for Loss
plt.subplot(1, 2, 1)
plt.plot(range(1,epoch+2),global_losses)
plt.title("Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
# plt.savefig("loss.png")
# plt.close()

plt.subplot(1, 2, 2)
plt.plot(range(1,epoch+2), global_train_acc, label='Training acc', marker='o')
plt.plot(range(1,epoch+2), global_test_acc, label='Testing acc', marker='o')
plt.title('Accuracy over epochs')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.grid()
plt.savefig("loss_and_acc.png")
plt.close()


from jax import jit

@jit
def update(params, x, y, epoch_number):
	loss_value, grads = value_and_grad(loss)(params, x, y)
	lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
	return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)], loss_value

@jit
def batch_accuracy(params, images, targets):
	images = jnp.reshape(images, (len(images), NUM_PIXELS))
	predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
	return jnp.mean(predicted_class == targets)

params = init_network_params(LAYER_SIZES, random.PRNGKey(0), scale=PARAM_SCALE)

delta_epochs = 0
global_losses = []
global_train_acc = []
global_test_acc = []
for epoch in range(num_epochs):
	start_time = time.time()
	losses = []

	for x, y in train_data:
		x = jnp.reshape(x, (len(x), NUM_PIXELS))
		y = one_hot(y, NUM_LABELS)
		params, loss_value = update(params, x, y, epoch)
		losses.append(loss_value)
	epoch_time = time.time() - start_time
	
	start_time = time.time()
	
	train_acc = accuracy(params, train_data)
	test_acc = accuracy(params, test_data)
	eval_time = time.time() - start_time
	
	print(f"Epoch {epoch}/{num_epochs} in {epoch_time:.2f} sec")
	print("Eval in {:0.2f} sec".format(eval_time))
	print("Training set loss {}".format(jnp.mean(jnp.array(losses))))
	print("Training set accuracy {}".format(train_acc))
	print("Test set accuracy {}\n".format(test_acc))

	global_losses.append(jnp.mean(jnp.array(losses)))
	global_train_acc.append(train_acc)
	global_test_acc.append(test_acc)

	if (test_acc - test_acc_after) < delta:
		delta_epochs += 1
		test_acc_after = test_acc
		print(f'delta_epochs = {delta_epochs}\n\n')
	else:
		delta_epochs = 0
		test_acc_after = test_acc
	if delta_epochs == EPOCH_STOP:
		print(f"early stopping, acc_test = {test_acc:.2f}\n\n")
		break

plt.subplot(1, 2, 1)
plt.plot(range(1,epoch+2),global_losses)
plt.title("Loss JIT")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
# plt.savefig("loss_jit.png")
# plt.close()

plt.subplot(1, 2, 2)
plt.plot(range(1,epoch+2), global_train_acc, label='Training acc', marker='o')
plt.plot(range(1,epoch+2), global_test_acc, label='Testing acc', marker='o')
plt.title('Accuracy JIT over epochs')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.grid()
plt.savefig("loss_and_acc_jit.png")


import pickle

model_weights_file = 'mlp_weights.pickle'

with open(model_weights_file, 'wb') as file:
	pickle.dump(params, file)

with open(model_weights_file, 'rb') as file:
	restored_params = pickle.load(file)

'''
>>> import jax
>>> jax.devices()
[CpuDevice(id=0)]


EEpoch 0/600 in 16.39 sec
Eval in 4.20 sec
Training set loss 0.4109187424182892
Training set accuracy 0.9291833639144897
Test set accuracy 0.930111825466156

Epoch 1/600 in 13.71 sec
Eval in 2.49 sec
Training set loss 0.3775194585323334
Training set accuracy 0.9490000009536743
Test set accuracy 0.9503793716430664

2025-02-05 09:11:11.129329: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 2/600 in 12.98 sec
Eval in 2.63 sec
Training set loss 0.3709490895271301
Training set accuracy 0.9595666527748108
Test set accuracy 0.9583665728569031

Epoch 3/600 in 12.95 sec
Eval in 2.36 sec
Training set loss 0.36744096875190735
Training set accuracy 0.9660333395004272
Test set accuracy 0.9630590677261353

2025-02-05 09:11:44.370033: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 4/600 in 12.41 sec
Eval in 2.53 sec
Training set loss 0.36510029435157776
Training set accuracy 0.9704499840736389
Test set accuracy 0.9673522114753723

Epoch 5/600 in 13.01 sec
Eval in 2.37 sec
Training set loss 0.3633940815925598
Training set accuracy 0.9739000201225281
Test set accuracy 0.9700478911399841

Epoch 6/600 in 12.48 sec
Eval in 3.11 sec
Training set loss 0.36207592487335205
Training set accuracy 0.9763666987419128
Test set accuracy 0.9719448685646057

Epoch 7/600 in 13.12 sec
Eval in 2.66 sec
Training set loss 0.36101675033569336
Training set accuracy 0.9783333539962769
Test set accuracy 0.973542332649231

Epoch 8/600 in 12.73 sec
Eval in 2.51 sec
Training set loss 0.36014029383659363
Training set accuracy 0.9801000356674194
Test set accuracy 0.9747403860092163

Epoch 9/600 in 12.77 sec
Eval in 2.39 sec
Training set loss 0.3593975305557251
Training set accuracy 0.9819000363349915
Test set accuracy 0.9751397371292114

delta_epochs = 1


2025-02-05 09:13:15.252745: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 10/600 in 12.52 sec
Eval in 3.51 sec
Training set loss 0.35875600576400757
Training set accuracy 0.9833500385284424
Test set accuracy 0.9764376878738403

Epoch 11/600 in 12.69 sec
Eval in 2.50 sec
Training set loss 0.35819345712661743
Training set accuracy 0.9846667051315308
Test set accuracy 0.9773362278938293

delta_epochs = 1


Epoch 12/600 in 13.28 sec
Eval in 2.35 sec
Training set loss 0.35769400000572205
Training set accuracy 0.9855000376701355
Test set accuracy 0.9774360656738281

delta_epochs = 2


Epoch 13/600 in 12.48 sec
Eval in 3.18 sec
Training set loss 0.3572457730770111
Training set accuracy 0.9862499833106995
Test set accuracy 0.9778354167938232

delta_epochs = 3


Epoch 14/600 in 12.72 sec
Eval in 2.94 sec
Training set loss 0.35683971643447876
Training set accuracy 0.9868833422660828
Test set accuracy 0.9785343408584595

delta_epochs = 4


Epoch 15/600 in 13.91 sec
Eval in 2.53 sec
Training set loss 0.35646894574165344
Training set accuracy 0.9874333739280701
Test set accuracy 0.9790335297584534

delta_epochs = 5


Epoch 16/600 in 13.19 sec
Eval in 2.90 sec
Training set loss 0.35612794756889343
Training set accuracy 0.9883000254631042
Test set accuracy 0.979632556438446

delta_epochs = 6


early stopping, acc_test = 0.98


Epoch 0/600 in 4.30 sec
Eval in 3.06 sec
Training set loss 0.4109187424182892
Training set accuracy 0.9291833639144897
Test set accuracy 0.930111825466156

delta_epochs = 1


Epoch 1/600 in 3.94 sec
Eval in 3.11 sec
Training set loss 0.3775194585323334
Training set accuracy 0.9490000009536743
Test set accuracy 0.9503793716430664

Epoch 2/600 in 3.94 sec
Eval in 2.91 sec
Training set loss 0.3709490895271301
Training set accuracy 0.9595666527748108
Test set accuracy 0.9583665728569031

2025-02-05 09:15:23.804843: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 3/600 in 3.98 sec
Eval in 2.96 sec
Training set loss 0.36744096875190735
Training set accuracy 0.9660333395004272
Test set accuracy 0.9630590677261353

Epoch 4/600 in 3.90 sec
Eval in 3.03 sec
Training set loss 0.36510029435157776
Training set accuracy 0.9704499840736389
Test set accuracy 0.9673522114753723

Epoch 5/600 in 3.85 sec
Eval in 2.98 sec
Training set loss 0.3633940815925598
Training set accuracy 0.9739000201225281
Test set accuracy 0.9700478911399841

Epoch 6/600 in 3.85 sec
Eval in 3.03 sec
Training set loss 0.36207592487335205
Training set accuracy 0.9763666987419128
Test set accuracy 0.9719448685646057

Epoch 7/600 in 3.84 sec
Eval in 3.01 sec
Training set loss 0.36101675033569336
Training set accuracy 0.9783499836921692
Test set accuracy 0.973542332649231

Epoch 8/600 in 4.03 sec
Eval in 3.02 sec
Training set loss 0.36014029383659363
Training set accuracy 0.9801000356674194
Test set accuracy 0.9747403860092163

Epoch 9/600 in 4.03 sec
Eval in 2.94 sec
Training set loss 0.3593975305557251
Training set accuracy 0.9819000363349915
Test set accuracy 0.9751397371292114

delta_epochs = 1


Epoch 10/600 in 4.00 sec
Eval in 2.99 sec
Training set loss 0.35875600576400757
Training set accuracy 0.9833500385284424
Test set accuracy 0.9764376878738403

Epoch 11/600 in 3.93 sec
Eval in 3.03 sec
Training set loss 0.35819342732429504
Training set accuracy 0.9846667051315308
Test set accuracy 0.9773362278938293

delta_epochs = 1


Epoch 12/600 in 4.07 sec
Eval in 2.98 sec
Training set loss 0.35769400000572205
Training set accuracy 0.9855000376701355
Test set accuracy 0.9774360656738281

delta_epochs = 2


Epoch 13/600 in 4.00 sec
Eval in 2.90 sec
Training set loss 0.3572457730770111
Training set accuracy 0.9862499833106995
Test set accuracy 0.9778354167938232

delta_epochs = 3


Epoch 14/600 in 3.95 sec
Eval in 3.08 sec
Training set loss 0.35683971643447876
Training set accuracy 0.9868833422660828
Test set accuracy 0.9785343408584595

delta_epochs = 4


Epoch 15/600 in 3.94 sec
Eval in 3.16 sec
Training set loss 0.35646894574165344
Training set accuracy 0.9874333739280701
Test set accuracy 0.9790335297584534

delta_epochs = 5


Epoch 16/600 in 3.98 sec
Eval in 3.03 sec
Training set loss 0.35612794756889343
Training set accuracy 0.9883000254631042
Test set accuracy 0.979632556438446

delta_epochs = 6


early stopping, acc_test = 0.98
'''