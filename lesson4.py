import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

# load image
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3])
top_left_slice = tf.slice(image, [0, 0, 0], [2000, 2000, 3])
bottom_left_slice = tf.slice(image, [3528, 0, 0], [2000, 2000, 3])
top_right_slice = tf.slice(image, [0, 1685, 0], [2000, 2000, 3])
bottom_right_slice = tf.slice(image, [3528, 1685, 0], [2000, 2000, 3])
greyscale_slice = tf.reduce_mean(image, 2)

with tf.Session() as session:
    print(raw_image_data.shape)
    greyscale_result = session.run(greyscale_slice, feed_dict={image: raw_image_data})

plt.imshow(greyscale_result)
plt.show()
