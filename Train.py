import os
import tensorflow as tf
import matplotlib.pyplot as plt

class InputTrainData:
    count = 0
    imagesize = 180
    batch = 128
    basedir = 'training_set/training_set/'
    dirs = os.listdir('training_set/training_set/')

def PathCount() :
    """Report how many training images exist per class directory."""
    # Track the total number of files across all subfolders.
    count = 0
    for dir in InputTrainData.dirs :
        # Grab every file inside the current class folder.
        files = list(os.listdir('training_set/training_set/' + dir))
        count = count + len(files)
    print(f'{dir} folder: {dir}, number of files: {len(files)}')
    print(f'Total number of files: {count}')
    return PathCount

class TrainData :
    train_ds = tf.keras.utils.image_dataset_from_directory(
        InputTrainData.basedir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(InputTrainData.imagesize, InputTrainData.imagesize),
        batch_size=InputTrainData.batch
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        InputTrainData.basedir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(InputTrainData.imagesize, InputTrainData.imagesize),
        batch_size=InputTrainData.batch
    )
    Animal_Names = train_ds.class_names
    num_classes = len(Animal_Names)
    i = 0



def SHDataset() :
    """Display a small sample of two training images and their labels."""
    plt.figure(figsize=(10,10))
    for images, labels in TrainData.train_ds.take(1):
        for i in range(2):
            plt.subplot(1, 2,  i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(TrainData.Animal_Names[labels[i]])
            plt.axis("off")
            plt.show()

    return SHDataset

class AugmentData :
    AutoTune = tf.data.AUTOTUNE
    train_ds = TrainData.train_ds.cache().shuffle(1000).prefetch(buffer_size=AutoTune)
    val_ds = TrainData.val_ds.cache().prefetch(buffer_size=AutoTune)

    #DataAugmentation Additional of Rotation, Zoom, Flip of Images
    DataAugmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", input_shape=(InputTrainData.imagesize, InputTrainData.imagesize, 3)),
        tf.keras.layers.RandomRotation(0.1), 
        tf.keras.layers.RandomZoom(0.1)
    ])

def VisualizeAugmentedData() :
    """Visualize nine augmented variants from the training dataset."""
    plt.figure(figsize=(10,10))
    for images, labels in AugmentData.train_ds.take(1):
        for i in range(9):
            # Apply the augmentation pipeline before each plot.
            images = AugmentData.DataAugmentation(images)
            plt.subplot(3, 3,  i + 1)
            plt.imshow(images[0].numpy().astype("uint8"))
            plt.title(TrainData.Animal_Names[labels[i]])
            plt.axis("off")
            plt.show()
    
#Create the Model
class ModelBuild :
    model = tf.keras.Sequential([
        AugmentData.DataAugmentation, 
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding = 'same', activation='relu'),
        tf.keras.layers.MaxPooling2D(), 
        tf.keras.layers.Conv2D(32, 3, padding = 'same', activation='relu'),
        tf.keras.layers.MaxPooling2D(), 
        tf.keras.layers.Conv2D(64, 3, padding = 'same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding = 'same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(TrainData.num_classes)
    ])

def TrainModel() :
    """Compile, train, and plot the CNN classifier performance."""
    ModelBuild.model.compile(optimizer='adam',
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

    ModelSummary = ModelBuild.model.summary()
    history = ModelBuild.model.fit(AugmentData.train_ds, epochs = 100, validation_data = AugmentData.val_ds) 
    history_acc = history.history['accuracy']
    history_val_acc = history.history['val_accuracy']
    epochs = range(len(history_acc))
    plt.figure()
    # Plot how accuracy evolves for both training and validation splits.
    plt.plot(epochs, history_acc, 'r', label='Training Accuracy')
    plt.plot(epochs, history_val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    return TrainModel

if __name__ == "__main__":
    PathCount()
    SHDataset()
    VisualizeAugmentedData()
    TrainModel()

