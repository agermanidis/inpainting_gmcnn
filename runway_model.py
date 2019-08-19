import numpy as np
import cv2
import os
import glob
import tensorflow as tf
from options.test_options import TestOptions
from util.util import generate_mask_rect, generate_mask_stroke
from net.network import GMCNNModel
import runway
from PIL import Image

config = TestOptions().parse()

g = tf.get_default_graph()
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False
sess = tf.InteractiveSession(graph=g, config=sess_config)

@runway.setup(options={'checkpoint_dir': runway.file(is_directory=True)})
def setup(opts):
    config.load_model_dir = opts['checkpoint_dir']
    config.dataset_name = os.path.basename(opts['checkpoint_dir'])
    model = GMCNNModel()
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])
    output = model.evaluate(input_image_tf, input_mask_tf, config=config, reuse=False)
    output = (output + 1) * 127.5
    output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
    output = tf.cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)), vars_list))
    sess.run(assign_ops)
    return output, input_image_tf, input_mask_tf

command_inputs = {
  'image': runway.image,
  'mask': runway.segmentation(label_to_id={'background': 0, 'mask': 1}, label_to_color={'background': [0, 0, 0], 'mask': [255, 255, 255]})
}

@runway.command('inpaint', inputs=command_inputs, outputs={'output': runway.image})
def inpaint(model, inputs):
    output, input_image_tf, input_mask_tf = model
    image = inputs['image']
    original_size = image.size
    image = np.array(image.resize((256, 256)), dtype=np.float32)
    mask = np.array(inputs['mask'].resize((256, 256)), dtype=np.float32)
    mask = np.expand_dims(mask, -1)
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    result = sess.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
    return Image.fromarray(result[0][:, :, ::-1]).resize(original_size)


if __name__ == "__main__":
    runway.run(model_options={'checkpoint_dir': 'checkpoints/places2_512x680_freeform'})