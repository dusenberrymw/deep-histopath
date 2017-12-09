from time import time
import math

import keras
from keras.models import load_model, Model
from keras.layers import Conv2D, Input
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
from train_mitoses import normalize
from preprocess_mitoses import gen_dense_coords, gen_patches
from breastcancer.inference import gen_batches#, surgery


def surgery(model):
  """Perform surgery to convert the model to a fully convolutional
  model that can be run over any input size.

  Args:
    model: A Keras convolutional model in which the final two layers
      consist of a flatten layer and a dense layer.

  Returns:
    A Keras model that has been surgically changed to a fully
    convolutional model.
  """
  # lazy importing to avoid premature GPU initialization
  from keras.models import Model
  from keras.layers import Conv2D, Input

  dense = model.layers[-1]
  pre_dense = model.layers[-3]

  # model surgery
  _, h, w, c = pre_dense.output_shape
  conv2d = Conv2D(1, (h, w))  # 1 filter of shape (h,w,c), stride 1, 0 pad
  x = pre_dense.output
  x = conv2d(x)
  model_all_conv = Model(inputs=model.inputs, outputs=x, name="model")

  # make the model size invariant
  inputs = Input(shape=(None, None, 3))
  outputs = model_all_conv(inputs)
  model_all_conv = Model(inputs=inputs, outputs=outputs, name="model")

  # weight surgery
  W, b = dense.get_weights()
  conv2d.set_weights([W.reshape(h,w,c,1), b])  # one filter

  return model_all_conv


# laptop
model_path = "experiments/best/171113_041432_vgg_patch_size_64_batch_size_32_clf_epochs_5_finetune_epochs_5_clf_lr_0.0010002776984729558_finetune_lr_0.00010008495232217907_finetune_momentum_0.8744471920298894_finetune_layers_-1_l2_0.0037563140329855663_aug_True_marg_False/checkpoints/0.74172_f1_1.7319_loss_8_epoch_model.hdf5"

# server
# - vgg
model_name = "vgg"
model_path = "experiments/mitoses/hyp_png3/171113_041432_vgg_patch_size_64_batch_size_32_clf_epochs_5_finetune_epochs_5_clf_lr_0.0010002776984729558_finetune_lr_0.00010008495232217907_finetune_momentum_0.8744471920298894_finetune_layers_-1_l2_0.0037563140329855663_aug_True_marg_False/checkpoints/0.74172_f1_1.7319_loss_8_epoch_model.hdf5"

# - resnet
model_name = "resnet"
model_path = "experiments/mitoses/hyp_png3_c/171118_203459_resnet_patch_size_64_batch_size_32_clf_epochs_5_finetune_epochs_5_clf_lr_0.0013461633314710202_finetune_lr_0.0010454540440176178_finetune_momentum_0.9482895329196482_finetune_layers_-1_l2_0.0032449804058511645_aug_True_marg_False/checkpoints/0.72938_f1_0.067459_loss_7_epoch_model.hdf5"

model = load_model(model_path)

model.summary()

model_all_conv = surgery(model)

model_all_conv.summary()

model_all_conv.layers[-1].summary()

save_path = model_path[:-5] + "_all_conv.hdf5"
model_all_conv.save(save_path, include_optimizer=False)



# test
# patch
# laptop
#img = Image.open("data/mitoses/patches/train/mitosis/1_03_02_450_1918_0_0_0.png")
#
## server
#img = Image.open("data/mitoses/patches_aug_strat_sampled_fp_oversampling_png_improved_gen_dense3/train/mitosis/1_01_31_1057_1795_0_0_0.png")

# whole image
img = Image.open("data/mitoses/mitoses_train_image_data/01/02.tif")  # (2000,2000,3)
print(img.size)

# random
img = np.random.randint(0, 256, size=(2000,2000,3), dtype=np.uint8)

img = np.random.randint(0, 256, size=(5657,5657,3), dtype=np.uint8)

img = np.random.randint(0, 256, size=(2657,2657,3), dtype=np.uint8)


# convert to array
x = np.array(img).astype(np.float32)


def predict_patches(x, model, model_name, patch_size, stride, batch_size):
  h, w, c = x.shape
  tile_indices = gen_dense_coords(h, w, patch_size, stride)
  tiles = (el[0] for el in gen_patches(x, tile_indices, patch_size, 0, 0, 0, 1))
  probs = np.empty((0, 1))
  tile_batches = gen_batches(tiles, batch_size, include_partial=True)
  for tile_batch in tile_batches:
    tile_stack = np.stack(tile_batch, axis=0)
    tile_stack = normalize((tile_stack / 255).astype(dtype=np.float32), model_name)
    prob_np = model.predict(tile_stack, batch_size)
    probs = np.concatenate((probs, prob_np), axis=0)
  return probs


def predict_surgery(x, model, model_name, patch_size, stride):
  if model_name in ("vgg", "vgg19"):
    model_stride = 32  # implicit stride of the fully-convolution version of this model
    #downsample =  # compute exact downsample based on max pooling layers
    max_size = 2000
  elif model_name == "resnet":
    model_stride = 64
    #downsample =  # compute exact downsample based on [strided] max pooling and conv layers
    max_size = 2000
  else:
    raise("unknown model!")
  assert model_stride % stride == 0
  x_norm = normalize(x/255, model_name)
  half_size = int(patch_size / 2)
  padding = ((half_size, half_size-1), (half_size, half_size-1)) + ((0, 0),) * (np.ndim(x)-2)
  #padding = ((half_size, 16), (half_size, 16)) + ((0, 0),) * (np.ndim(x)-2)
  x_norm_padded = np.pad(x_norm, padding, 'reflect')
  h, w, _ = x_norm_padded.shape
  hout = math.floor((h-patch_size)/stride + 1)
  wout = math.floor((w-patch_size)/stride + 1)
  probs = np.zeros((hout, wout), dtype=np.float32) - 1
  out_stride = int(model_stride/stride)
  for i in range(out_stride):
    for j in range(out_stride):
      # 1. slide image to right by stride pixels
      # 2. compute preds
      # 3. assign to out_all_conv_padded starting at i, strided by model_stride/stride
      x_subset = x_norm_padded[i*stride:, j*stride:, :]
      out = model_all_conv.predict_on_batch(np.expand_dims(x_subset, 0))[0].squeeze()
      probs[i::out_stride, j::out_stride] = out
  return probs


def predict_surgery_v2(x, model, model_name, patch_size, stride):
  if model_name in ("vgg", "vgg19"):
    model_stride = 32  # implicit stride of the fully-convolution version of this model
    #downsample =  # compute exact downsample based on max pooling layers
    max_size = 2000
  elif model_name == "resnet":
    model_stride = 64
    #downsample =  # compute exact downsample based on [strided] max pooling and conv layers
    max_size = 2000
  else:
    raise("unknown model!")
  assert model_stride % stride == 0
  x_norm = normalize(x/255, model_name)
  half_size = int(patch_size / 2)
  padding = ((half_size, half_size-1), (half_size, half_size-1)) + ((0, 0),) * (np.ndim(x)-2)
  #padding = ((half_size, 16), (half_size, 16)) + ((0, 0),) * (np.ndim(x)-2)
  x_norm_padded = np.pad(x_norm, padding, 'reflect')
  hpad, wpad, _ = x_norm_padded.shape
  hout = math.floor((hpad-patch_size)/stride + 1)
  wout = math.floor((wpad-patch_size)/stride + 1)
  hpad_max = wpad_max = max_size + half_size + half_size-1
  hout_max = math.floor((hpad_max-patch_size)/stride + 1)
  wout_max = math.floor((wpad_max-patch_size)/stride + 1)
  probs = np.zeros((hout, wout), dtype=np.float32) - 1
  out_stride = int(model_stride/stride)
  # cut up image if necessary
  for hi, hlow in enumerate(range(0, hpad-half_size, max_size+half_size)):
    for wi, wlow in enumerate(range(0, wpad-half_size, max_size+half_size)):
      # stride over image for stride < model_stride
      for i in range(out_stride):
        for j in range(out_stride):
          print(hi, wi, i, j)
          # 1. slide image to right by stride pixels
          # 2. compute preds
          # 3. assign to out_all_conv_padded starting at i, strided by model_stride/stride
          x_subset = x_norm_padded[hlow+i*stride:hlow+hpad_max, wlow+j*stride:wlow+wpad_max, :]
          hsub, wsub, _ = x_subset.shape
          sub_padding = ((0, hpad_max-hsub), (0, wpad_max-wsub)) + ((0, 0),) * (np.ndim(x)-2)
          x_subset_padded = np.pad(x_subset, sub_padding, 'reflect')
          out = model_all_conv.predict_on_batch(np.expand_dims(x_subset_padded, 0))[0].squeeze()
          hout_sub, wout_sub = probs[hi*hout_max+i:hi*hout_max+hout_max:out_stride, wi*wout_max+j:wi*wout_max+wout_max:out_stride].shape
          probs[hi*hout_max+i:hi*hout_max+hout_max:out_stride, wi*wout_max+j:wi*wout_max+wout_max:out_stride] = out[:hout_sub, :wout_sub]
  return probs


stride = 16 #1 #8 #4 #16 #32 #64
patch_size = 64

# patches approach
start = time()
batch_size = 256
preds = predict_patches(x, model, model_name, patch_size, stride, batch_size)
end = time()
print(f"patches approach: {end-start} secs")

# surgery approach
start = time()
out_all_conv_padded = predict_surgery_v2(x, model, model_name, patch_size, stride)
end = time()
print(f"surgery approach: {end-start} secs")

## surgery v2 approach
#start = time()
#out_all_conv_padded_v2 = predict_surgery_v2(x, model, model_name, patch_size, stride)
#end = time()
#print(f"surgery v2 approach: {end-start} secs")

#assert np.allclose(out_all_conv_padded, out_all_conv_padded_v2)


preds = preds.reshape(out_all_conv_padded.shape)

print(preds.shape)
print(out_all_conv_padded.shape)

#print(out_all_conv_padded_v2.shape)

#assert np.allclose(preds, out_all_conv_padded)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

spreds = sigmoid(preds)
sout_all_conv_padded = sigmoid(out_all_conv_padded)
print(spreds.shape)
print(sout_all_conv_padded.shape)

#sout_all_conv_padded_v2 = sigmoid(out_all_conv_padded_v2)
#print(sout_all_conv_padded_v2.shape)

#assert np.allclose(spreds, sout_all_conv_padded)

diff = abs(preds - out_all_conv_padded)
print(diff.shape)
print(np.max(diff))
print()

#diff_v2 = abs(out_all_conv_padded - out_all_conv_padded_v2)
#print(diff_v2.shape)
#print(np.max(diff_v2))
#print()

sdiff = abs(spreds - sout_all_conv_padded)
print(sdiff.shape)
print(np.max(sdiff))
print()

#sdiff_v2 = abs(sout_all_conv_padded - sout_all_conv_padded_v2)
#print(sdiff_v2.shape)
#print(np.max(sdiff_v2))
#print()

print(sigmoid(preds)[(sdiff > 0).nonzero()])
print(sigmoid(out_all_conv_padded)[(sdiff > 0).nonzero()])
print(sdiff)
print()

print(np.count_nonzero(sdiff > 0))
print(np.count_nonzero(sdiff > 0.1))
print(np.count_nonzero(sdiff > 0.2))
print(np.count_nonzero(sdiff > 0.3))
print(np.count_nonzero(sdiff > 0.4))
print(np.count_nonzero(sdiff > 0.5))
print(np.count_nonzero(sdiff > 0.6))
print()

#print(np.count_nonzero(sdiff_v2 > 0))
#print(np.count_nonzero(sdiff_v2 > 0.1))
#print(np.count_nonzero(sdiff_v2 > 0.2))
#print(np.count_nonzero(sdiff_v2 > 0.3))
#print(np.count_nonzero(sdiff_v2 > 0.4))
#print(np.count_nonzero(sdiff_v2 > 0.5))
#print(np.count_nonzero(sdiff_v2 > 0.6))
#print()

changed = np.logical_and(spreds > 0.5, sout_all_conv_padded <= 0.5)
print(np.count_nonzero(changed))
print(changed.nonzero())
print(spreds[changed])
print(sout_all_conv_padded[changed])
print()

changed = np.logical_and(spreds <= 0.5, sout_all_conv_padded > 0.5)
print(np.count_nonzero(changed))
print(changed.nonzero())
print(spreds[changed])
print(sout_all_conv_padded[changed])

# original vgg base maps (64,64,3) volumes to (2,2,512) volumes, i.e., downsample by 32, and then
# a 2x2, stride 1, 0 pad conv2d op will map to (1,1,1)
# (2000,2000,3) will map to (62,62,512), and then conv2d op will map to (61,61,1) due to
# (Hin - Hf + 2*pad)/stride + 1 = (62 - 2 + 2*0)/1 + 1 = 61


def test_model_surgery():
  from keras.layers import Input
  from train_mitoses import normalize
  from train_mitoses import create_model

  def test1(model_name):
    images_dummy = Input(shape=(64,64,3))
    model, _ = create_model(model_name, (64,64,3), images_dummy)

    # surgery
    model_all_conv = surgery(model)

    img = np.random.randint(0, 256, size=(64,64,3), dtype=np.uint8)
    x = np.array(img).astype(np.float32)
    x_norm = normalize(x/255, model_name)

    out = model.predict_on_batch(np.expand_dims(x_norm, 0))[0]
    out_all_conv = model_all_conv.predict_on_batch(np.expand_dims(x_norm, 0))[0]

    assert np.allclose(out, out_all_conv)

  test1("vgg")
  test1("resnet")



# performance
# - 5657x5657 input:
#   - vgg
#     - patches approach: 159.98368549346924 secs
#     - surgery approach: 122.21448707580566 secs
#   - resnet
#     - patches approach: 413.63410663604736 secs
#     - surgery approach: 601.5547530651093 secs


