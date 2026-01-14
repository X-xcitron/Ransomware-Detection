import os
import tensorflow as tf

# --------- Paths (edit if needed) ---------
OUTPUT_DIR = 'D:/Ransomware-Detection-using-Deep-Learning-master/output'
TFRECORD_PATTERN = os.path.join(OUTPUT_DIR, 'train-*-of-*.tfrecord')
LABELS_FILE = 'D:\Ransomware-Detection-using-Deep-Learning-master\label.txt'

# --------- Hyperparameters ---------
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 6
VAL_SPLIT_MOD = 10   # 1/VAL_SPLIT_MOD ~ validation fraction (10% here)
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# --------- Labels / classes ---------
with open(LABELS_FILE, 'r') as f:
    CLASS_NAMES = [ln.strip() for ln in f.readlines() if ln.strip()]
NUM_CLASSES = len(CLASS_NAMES)
IS_BINARY = (NUM_CLASSES == 2)

# --------- TFRecord parsing ---------
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/height':  tf.io.FixedLenFeature([], tf.int64),
    'image/width':   tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized):
    example = tf.io.parse_single_example(serialized, feature_description)
    img_bytes = example['image/encoded']
    label = tf.cast(example['image/class/label'], tf.int32)

    # Decode as JPEG (your preprocessor stored JPEG bytes)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    # Per-image standardization similar to your original pipeline
    img = tf.image.per_image_standardization(img)
    return img, label

def make_dataset(pattern, for_training=True):
    files = tf.data.Dataset.list_files(pattern, shuffle=True, seed=SEED)
    ds = files.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type=None),
        cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)

    # Split train/val by index to avoid needing two separate folders
    enum = ds.enumerate()
    if for_training:
        enum = enum.filter(lambda i, _: (i % VAL_SPLIT_MOD) > 0)
    else:
        enum = enum.filter(lambda i, _: (i % VAL_SPLIT_MOD) == 0)
    ds = enum.map(lambda _, x: x, num_parallel_calls=AUTOTUNE)

    if for_training:
        ds = ds.shuffle(8 * BATCH_SIZE, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(TFRECORD_PATTERN, for_training=True)
val_ds   = make_dataset(TFRECORD_PATTERN, for_training=False)

# --------- Model (Keras CNN) ---------
def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

        tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(128, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(256, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(256, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),

        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid') if IS_BINARY
        else tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    if IS_BINARY:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                   tf.keras.metrics.AUC(name='auc')]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=metrics
    )
    return model

model = build_model(NUM_CLASSES)
model.summary()

# --------- Training ---------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'model_checkpoint.keras'),
        save_best_only=True, monitor='val_accuracy', mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --------- Save final model ---------
model.save(os.path.join(OUTPUT_DIR, 'malware_cnn_final.keras'))

# --------- Optional evaluation ---------
eval_results = model.evaluate(val_ds)
print('Validation results:', eval_results)
