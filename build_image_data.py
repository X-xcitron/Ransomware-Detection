from __future__ import absolute_import, division, print_function
import os
import random
import sys
import threading
from datetime import datetime
import numpy as np
import tensorflow as tf

# Manual config
train_directory = 'D:/Malware-Detection-using-Deep-Learning-master/Dataset'
output_directory = 'D:/Malware-Detection-using-Deep-Learning-master/output'
labels_file = 'D:/Malware-Detection-using-Deep-Learning-master/label.txt'
validation_directory = None
train_shards = 2
validation_shards = 0
num_threads = 2

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, label, text, height, width):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example

def png_to_jpeg(image_data):
    image = tf.image.decode_png(image_data, channels=3)
    image_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
    return image_jpeg.numpy()

def decode_jpeg(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image_np = image.numpy()
    assert len(image_np.shape) == 3
    assert image_np.shape[2] == 3
    return image_np

def _is_png(filename):
    return '.png' in filename

def _process_image(filename):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = png_to_jpeg(image_data)
    image = decode_jpeg(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width

def _process_image_files_batch(thread_index, ranges, name, filenames, texts, labels, num_shards):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d.tfrecord' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]
            image_buffer, height, width = _process_image(filename)
            example = _convert_to_example(filename, image_buffer, label, text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' % (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' % (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, filenames, texts, labels, num_shards):
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int32)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()
    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, filenames, texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(data_dir, labels_file):
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in open(labels_file, 'r').readlines()]
    labels, filenames, texts = [], [], []
    label_index = 0
    for text in unique_labels:
        jpeg_file_path = os.path.join(data_dir, text, '*')
        matching_files = tf.io.gfile.glob(jpeg_file_path)
        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)
        label_index += 1
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))
    return filenames, texts, labels

def _process_dataset(name, directory, num_shards, labels_file):
    if directory is None or num_shards == 0:
        return
    filenames, texts, labels = _find_image_files(directory, labels_file)
    _process_image_files(name, filenames, texts, labels, num_shards)

def main():
    print('Saving results to %s' % output_directory)
    if validation_directory and os.path.exists(validation_directory) and validation_shards > 0:
        _process_dataset('validation', validation_directory, validation_shards, labels_file)
    _process_dataset('train', train_directory, train_shards, labels_file)

if __name__ == '__main__':
    main()
