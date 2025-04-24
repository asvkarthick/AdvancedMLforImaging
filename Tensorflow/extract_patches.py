import tensorflow as tf

n = 10
# images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100 in order
images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

# We generate four outputs as follows:
# 1. 3x3 patches with stride length 5
# 2. Same as above, but the rate is increased to 2
# 3. 4x4 patches with stride length 7; only one patch should be generated
# 4. Same as above, but with padding set to 'SAME'
# with tf.Session() as sess:
print(tf.image.extract_patches(images=images, sizes=[1, 3, 3, 1], strides=[1, 5, 5, 1], rates=[1, 1, 1, 1], padding='VALID'), '\n\n')
print(tf.image.extract_patches(images=images, sizes=[1, 3, 3, 1], strides=[1, 5, 5, 1], rates=[1, 2, 2, 1], padding='VALID'), '\n\n')
print(tf.image.extract_patches(images=images, sizes=[1, 4, 4, 1], strides=[1, 7, 7, 1], rates=[1, 1, 1, 1], padding='VALID'), '\n\n')
print(tf.image.extract_patches(images=images, sizes=[1, 4, 4, 1], strides=[1, 7, 7, 1], rates=[1, 1, 1, 1], padding='SAME'), '\n\n')


print('Image : ', images)
print('Image shape : ', tf.shape(images))
