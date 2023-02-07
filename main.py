import os
import shutil
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from model import model
from keras.preprocessing import image

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot(data, txt):
    plt.subplot(1, 2, 1)
    plt.plot(data.history['accuracy'])
    plt.plot(data.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(data.history['loss'])
    plt.plot(data.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    _txt = f"Kernel:{txt[0]}x{txt[0]}, pool:{txt[1]}x{txt[1]}, " \
           f"padding:{txt[2]}"
    plt.figtext(0.5, 0.01, _txt, wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig(img + 'k{}-p{}-{}'.format(txt[0], txt[1], txt[2]) + '.png')
    # plt.show()


def train(k, p, padd):
    _model = model(k, p, padd)

    train_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_dataset = image.ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=BS,
        class_mode='binary'
    )
    validation_generator = test_dataset.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        batch_size=BS,
        class_mode='binary'
    )

    hist = _model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True),
                   tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)]
    )

    print(
        f"\nAccuracy: {round(hist.history['accuracy'][-1], 3)}, Val_accuracy: {round(hist.history['val_accuracy'][-1], 3)}\n")
    file = open("log.txt", "a")
    file.write(f"Kernel:{k}, pool:{p}, padding:{padd} - Accuracy: {round(hist.history['accuracy'][-1], 3)}, "
               f"Val_accuracy: {round(hist.history['val_accuracy'][-1], 3)}\n")
    file.close()
    plot(data=hist, txt=[k, p, padd])


def load():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.mkdir(DATASET_DIR)
    os.mkdir(TRAIN_DIR)
    os.mkdir(VAL_DIR)
    os.mkdir(TARGET_COVID_DIR)
    os.mkdir(TARGET_NORMAL_DIR)
    os.mkdir(VAL_COVID_DIR)
    os.mkdir(VAL_NORMAL_DIR)
    if os.path.exists(AUG_SAVE_PATH):
        shutil.rmtree(AUG_SAVE_PATH)
    os.mkdir(AUG_SAVE_PATH)
    os.mkdir(TRAIN_AUG_SAVE_PATH)
    os.mkdir(VAL_AUG_SAVE_PATH)
    # Copy COVID-19 images with view point PA from Downloaded directory to Target Directory
    df = pd.read_csv(FILE_PATH)
    PA_images = df.query("finding == 'Pneumonia/Viral/COVID-19' and view == 'PA'")["filename"]
    div1 = int(RATIO * len(PA_images))
    train_PA_images = PA_images[:div1]
    val_PA_images = PA_images[div1:]
    for image_name in train_PA_images:
        image_path = os.path.join(COVID_PATH, image_name)
        target_path = os.path.join(TARGET_COVID_DIR, image_name)
        try:
            shutil.copy2(image_path, target_path)
        except FileNotFoundError:
            print(f"Invalid File: {image_name}")
    print(f"Total no.of covid images in training data: {len(os.listdir(TARGET_COVID_DIR))}")
    for image_name in val_PA_images:
        image_path = os.path.join(COVID_PATH, image_name)
        target_path = os.path.join(VAL_COVID_DIR, image_name)
        try:
            shutil.copy2(image_path, target_path)
        except FileNotFoundError:
            print(f"Invalid File: {image_name}")
    print(f"Total no.of covid images in validation data: {len(os.listdir(VAL_COVID_DIR))}\n")

    # random.shuffle(image_names)  # it will randomly shuffle names in list
    normal_images = os.listdir(NORMAL_PATH)
    div2 = int(RATIO * len(normal_images))
    train_normal_images = normal_images[:div2]
    for image_name in train_normal_images:
        image_path = os.path.join(NORMAL_PATH, image_name)
        target_path = os.path.join(TARGET_NORMAL_DIR, image_name)
        try:
            shutil.copy2(image_path, target_path)
        except FileNotFoundError:
            print(f"Invalid File: {image_name}")
    print(f"Total no.of normal images in training data: {len(os.listdir(TARGET_NORMAL_DIR))}")
    val_normal_images = normal_images[div2:]
    for image_name in val_normal_images:
        image_path = os.path.join(NORMAL_PATH, image_name)
        target_path = os.path.join(VAL_NORMAL_DIR, image_name)
        try:
            shutil.copy2(image_path, target_path)
        except FileNotFoundError:
            print(f"Invalid File: {image_name}")
    print(f"Total no.of normal images in validation data: {len(os.listdir(VAL_NORMAL_DIR))}\n")


def main():
    for KERNEL in [2, 3, 4, 5, 6, 7]:
        for POOL in [2, 3, 4, 5]:
            for PAD in ["same", "valid"]:
                try:
                    load()
                    train(KERNEL, POOL, PAD)
                except Exception as _e:
                    file = open("log.txt", "a")
                    file.write(
                        f"Kernel:{KERNEL}, pool:{POOL}, padding:{PAD} - {str(_e)}\n")
                    file.close()


if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # logical_gpus = tf.config.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    RATIO = 0.9
    BS = 32
    EPOCHS = 20

    FILE_PATH = "./covid-chestxray-dataset-master/metadata.csv"
    COVID_PATH = "./covid-chestxray-dataset-master/images"
    NORMAL_PATH = "./chest_xray_normal/"
    DATASET_DIR = './Dataset'
    TRAIN_DIR = './Dataset/Train'
    VAL_DIR = './Dataset/Val'
    TARGET_COVID_DIR = "./Dataset/Train/Covid/"
    TARGET_NORMAL_DIR = "./Dataset/Train/Normal/"
    VAL_COVID_DIR = "./Dataset/Val/Covid/"
    VAL_NORMAL_DIR = "./Dataset/Val/Normal/"
    AUG_SAVE_PATH = "./aug/"
    TRAIN_AUG_SAVE_PATH = "./aug/train"
    VAL_AUG_SAVE_PATH = "./aug/val"
    img = './Images/'

    main()
