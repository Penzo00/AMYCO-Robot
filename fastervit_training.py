def fastervit_training():
    import os
    import warnings
    import numpy as np
    import random
    import tensorflow as tf
    from tensorflow import keras as tfk
    from tensorflow.keras import layers as tfkl
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import mixed_precision
    import json
    from keras_cv_attention_models import fastervit

    # Fix randomness
    seed = 42
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    # Suppress warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=Warning)

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # Verify TensorFlow GPU setup
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"CUDA Version: {tf.sysconfig.get_build_info()['cuda_version']}")
    print(f"CuDNN Version: {tf.sysconfig.get_build_info()['cudnn_version']}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU configured correctly.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found.")

    # Mixed Precision
    mixed_precision.set_global_policy('mixed_float16')

    # Load datasets
    def load_dataset(data_dir, batch_size, image_size):
        return tfk.preprocessing.image_dataset_from_directory(
            data_dir,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            label_mode='int',
            seed=seed
        )

    data_dir_train = "fastervit2_training/MO13k/train"
    data_dir_val = "fastervit2_training/MO13k/val"
    batch_size = 128
    image_size = (224, 224)

    train_ds = load_dataset(data_dir_train, batch_size, image_size)
    val_ds = load_dataset(data_dir_val, batch_size, image_size)

    # Save class names
    np.save('classes.npy', train_ds.class_names)

    # Compute class weights
    labels = np.concatenate([y for _, y in train_ds], axis=0)
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(labels), y=labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    with open('class_weight_dict.json', 'w') as f:
        json.dump(class_weights_dict, f)

    # Normalize datasets
    normalization_layer = tfk.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Load pre-trained FasterViT
    mm = fastervit.FasterViT0(pretrained="imagenet", input_shape=(224, 224, 3))
    mm = tfk.Model(inputs=mm.input, outputs=mm.layers[-3].output)
    mm.trainable = True

    # Freeze BatchNormalization layers
    for layer in mm.layers:
        if isinstance(layer, tfkl.BatchNormalization):
            layer.trainable = False

    def build_fastervitv0():
        inputs = tfk.Input(shape=(224, 224, 3))
        x = mm(inputs)
        x = tfkl.GlobalAveragePooling2D(name="avg_pool")(x)
        x = tfkl.Dense(2048, activation='swish', kernel_regularizer=tfk.regularizers.OrthogonalRegularizer(0.001))(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.Dense(1024, activation='swish', kernel_regularizer=tfk.regularizers.L1L2(0.001, 0.001))(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.Dropout(0.2)(x)
        outputs = tfkl.Dense(len(class_weights_dict), activation="softmax", dtype='float32')(x)

        model = tfk.Model(inputs=inputs, outputs=outputs, name='fastervit_model')

        lr_schedule = tfk.optimizers.schedules.CosineDecay(
            initial_learning_rate=2.5e-5, decay_steps=10 * len(train_ds), alpha=0.0
        )

        model.compile(
            optimizer=tfk.optimizers.Lion(
                learning_rate=lr_schedule, beta_1=0.95, beta_2=0.98, weight_decay=0.4
            ),
            loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                tfk.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tfk.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")
            ]
        )
        return model

    # Build and train model
    model = build_fastervitv0()
    model.summary()

    # Callbacks
    callbacks = [
        tfk.callbacks.ModelCheckpoint(
            filepath="checkpoints/cp-{epoch:04d}.keras", save_best_only=True
        ),
        tfk.callbacks.TensorBoard(log_dir='./logs'),
        tfk.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-4),
        tfk.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tfk.callbacks.CSVLogger('training.log', append=True)
    ]

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # Save the trained model
    tf.saved_model.save(model, "amyco_model")
fastervit_training()