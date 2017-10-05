'''
Use a pre-trained ImageNet model
to identify  Simpsons characters.
'''
import os
from pprint import pprint

import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from scipy.misc import imread
import numpy as np

num_classes = 47

character_names = [
    "apu_nahasapeemapetilon",
    "professor_john_frink",
    "charles_montgomery_burns",
    "otto_mann",
    "groundskeeper_willie",
    "lisa_simpson",
    "bart_simpson",
    "snake_jailbird",
    "chief_wiggum",
    "comic_book_guy",
    "carl_carlson",
    "homer_simpson",
    "agnes_skinner",
    "lenny_leonard",
    "jimbo_jones",
    "martin_prince",
    "ned_flanders",
    "waylon_smithers",
    "sideshow_mel",
    "principal_skinner",
    "gil",
    "moe_szyslak",
    "maggie_simpson",
    "hans_moleman",
    "cletus_spuckler",
    "milhouse_van_houten",
    "nelson_muntz",
    "abraham_grampa_simpson",
    "barney_gumble",
    "bumblebee_man",
    "rainier_wolfcastle",
    "patty_bouvier",
    "troy_mcclure",
    "mayor_quimby",
    "fat_tony",
    "kent_brockman",
    "jasper_beardly",
    "selma_bouvier",
    "krusty_the_clown",
    "marge_simpson",
    "ralph_wiggum",
    "disco_stu",
    "miss_hoover",
    "sideshow_bob",
    "lionel_hutz",
    "helen_lovejoy",
    "edna_krabappel",
]

IMAGE_DIMS = (256, 256)

def extract_data():
    dataset_base = '/home/reece/innolitics/10x/datasets/simpsons'
    dataset = {}
    for data_dir, _, filenames in os.walk(dataset_base):
        image_data = (imread(os.path.join(data_dir, filename))
                      for filename in filenames
                      if filename.endswith('.jpg'))
        class_name = data_dir.split('/')[-1]
        dataset[class_name] = image_data
    return dataset

def training_data():
    dataset_base = '/home/reece/innolitics/10x/datasets/simpsons'
    for data_dir, _, filenames in os.walk(dataset_base):
        class_name = data_dir.split('/')[-1]
        if class_name != '':
            for idx, filename in enumerate(filenames):
                if idx % 5 != 0 and filename.endswith('.jpg'):
                    img = image.load_img(os.path.join(data_dir, filename), target_size=IMAGE_DIMS)
                    image_data = image.img_to_array(img)
                    image_data = np.expand_dims(image_data, axis=0)
                    image_data = preprocess_input(image_data)
                    out_data = np.zeros(len(character_names))
                    out_data[character_names.index(class_name)] = 1
                    out_data = np.expand_dims(out_data, axis=0) # Change this to have more samples per batch
                    yield (image_data, out_data)

def testing_data():
    dataset_base = '/home/reece/innolitics/10x/datasets/simpsons'
    for data_dir, _, filenames in os.walk(dataset_base):
        class_name = data_dir.split('/')[-1]
        if class_name != '':
            for idx, filename in enumerate(filenames):
                if idx % 5 == 0 and filename.endswith('.jpg'):
                    img = image.load_img(os.path.join(data_dir, filename), target_size=IMAGE_DIMS)
                    image_data = image.img_to_array(img)
                    image_data = np.expand_dims(image_data, axis=0)
                    image_data = preprocess_input(image_data)
                    out_data = np.zeros(len(character_names))
                    out_data[character_names.index(class_name)] = 1
                    out_data = np.expand_dims(out_data, axis=0) # Change this to have more samples per batch
                    yield (image_data, out_data)

def generate_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*IMAGE_DIMS, 3))

    x = base_model.output
    if x.shape.ndims > 2:
        x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(character_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    print("Compiling Model")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    # Now the top layers we added are well trained
    # And we can start tweaking the base model's top layers (carefully).

if __name__ == '__main__':
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='tensorboard', histogram_freq=0, write_graph=True, write_images=False)
    save_model_callback = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}.h5', verbose=3, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    model = generate_model()

    print("Training model")
    model.fit_generator(
        training_data(),
        4000,
        epochs=2,
        callbacks=[save_model_callback, tensorboard_callback],
        validation_data=testing_data(),
        validation_steps=900
    )

    print("Training Complete")

    score = model.evaluate_generator(testing_data(), 900)
    pprint(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
