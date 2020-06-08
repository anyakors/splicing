import keras
import keras.backend as K

import tensorflow as tf
#tf.enable_v2_behavior()
#tf.enable_eager_execution()
print('eagerly?', tf.executing_eagerly())

import numpy as np

def hot_encode_seq(let):
    if let=='A':
        return([1,0,0,0])
    elif let=='T':
        return([0,1,0,0])
    elif let=='C':
        return([0,0,1,0])
    elif let=='G':
        return([0,0,0,1])
    elif let=='O':
        return([0,0,0,0])

def hot_encode_label(let):
    if let=='p':
        return([0,0,0])
    elif let=='b':
        return([1,0,0])
    elif let=='a':
        return([0,1,0])
    elif let=='d':
        return([0,0,1])

def celoss_math(labels_flat, pred):
    pred = tf.transpose(tf.keras.backend.log(pred))
    labels_flat = tf.cast(labels_flat, pred.dtype)
    # labels_b.shape[0] = N (batch size), labels_b.shape[1] = 5000
    labels_flat = tf.reshape(labels_flat, [labels_flat.shape[0], labels_flat.shape[1]])
    # reshaping with dims 1, 2 because we alrd transposed it so it has dims [5000, N] now
    pred = tf.reshape(pred, [pred.shape[1], pred.shape[2]])
    return tf.math.negative(tf.math.reduce_sum(tf.linalg.diag(tf.keras.backend.dot(labels_flat, pred))))


transcripts_ = np.loadtxt('./transcripts_chunks_chr1', dtype='str', delimiter='\t', max_rows=10)
labels_ = np.loadtxt('./labels_chunks_chr1', dtype='str', delimiter='\t', max_rows=10)

transcripts = []
labels = []

# hot-encode
for i in range(len(transcripts_)):
    # hot-encode seq
    transcripts.append([np.array(hot_encode_seq(let)) for let in transcripts_[i]])
    # hot-encode labels
    labels.append([np.array(hot_encode_label(x)) for x in labels_[i]])


transcripts = np.array(transcripts)
labels = np.array(labels)

print(np.shape(transcripts), np.shape(labels))

x = tf.constant(transcripts)
y = tf.constant(labels)

mask_b = np.array([True, False, False])
mask_a = np.array([False, True, False])
mask_d = np.array([False, False, True])

labels_b = tf.boolean_mask(y, mask_b, axis=2)
n_b = tf.math.count_nonzero(labels_b)

labels_a = tf.boolean_mask(y, mask_a, axis=2)
n_a = tf.math.count_nonzero(labels_a)

labels_d = tf.boolean_mask(y, mask_d, axis=2)
n_d = tf.math.count_nonzero(labels_d)

print("Number of labels blank, a/d", n_b, n_a, n_d)

y_pred = tf.random.uniform(tf.shape(y), minval=0.001, maxval=0.999, dtype=tf.dtypes.float32)

pred_b = tf.boolean_mask(y_pred, mask_b, axis=2)
pred_a = tf.boolean_mask(y_pred, mask_a, axis=2)
pred_d = tf.boolean_mask(y_pred, mask_d, axis=2)

# now labels_b and pred_b should be of (N, 5000) shape

s_b = celoss_math(labels_b, pred_b)
s_a = celoss_math(labels_a, pred_a)
s_d = celoss_math(labels_d, pred_d)

print("Losses components:", tf.math.divide(s_b, tf.cast(n_b, s_b.dtype)), tf.math.divide(s_a, tf.cast(n_a, s_a.dtype)), tf.math.divide(s_d, tf.cast(n_d, s_d.dtype)))
#print("Overall loss:", tf.add())

