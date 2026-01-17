from Train import TrainData, ModelBuild
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def TestData():
    input_img = tf.keras.utils.load_img('test_set/test_set/cats/cat.4972.jpg', target_size=(180, 180))
    input_img_array = tf.keras.utils.img_to_array(input_img)
    input_img_exp_dim = tf.expand_dims(input_img_array, axis=0)  # Create a batch
    predictions = ModelBuild.model.predict(input_img_exp_dim)
    result = tf.nn.softmax(predictions[0])
    Predicted_Class = TrainData.Animal_Names[np.argmax(result)]
    Confidence = 100 * np.max(result)
    print(f'Predicted Class: {Predicted_Class}, Confidence: {Confidence:.2f}%')
    plt.imshow(input_img_array.astype("uint8"))
    plt.title(f'Predicted: {Predicted_Class} ({Confidence:.2f}%)')
    plt.axis("off")
    plt.show()
    return TestData

if __name__ == "__main__":
    TestData()

