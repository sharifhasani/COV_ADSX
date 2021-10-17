from keras.applications.densenet import DenseNet169
from CovidDetection.settings import MEDIA_ROOT
from os.path import join
import pickle
from cv2 import imread, resize
import numpy as np
from xgboost import XGBClassifier
import xgboost

def predict(image_name):
    image_path = join(MEDIA_ROOT, "uploads", "x-ray", image_name)
    img = imread(image_path)
    img = resize(img, (224, 224))
    img_array = np.array([img])
    pre_trained_model = load_DNN()
    feature = pre_trained_model.predict(img_array)
    XGBoost_mdoel = load_XGBoost_model()
    res = XGBoost_mdoel.predict(feature)
    save_and_display_gradcam(image_path, prepare_image(image_path), cam_path=join(MEDIA_ROOT, "heatmap", image_name))
    return res


def load_DNN():
    pre_trained_model = DenseNet169(include_top=False, input_shape=(224, 224, 3), pooling="avg")
    return pre_trained_model

def load_XGBoost_model():
    model_path = join(MEDIA_ROOT, "MLModel", "xgb_model.pkl")
    xgb_model = pickle.load(open(model_path, "rb"))
    return xgb_model

# heatmap 
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def build_model():
    decode_predictions = keras.applications.densenet.decode_predictions



def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def prepare_image(image_path):
    model_builder = DenseNet169
    img_size = (224, 224)
    preprocess_input = keras.applications.densenet.preprocess_input
    last_conv_layer_name = "conv5_block32_concat"

    # Prepare image
    img_array = preprocess_input(get_img_array(image_path, size=img_size))
    # print(plt.imshow(img_array[0]))


    # Make model
    model = model_builder(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    preds = model.predict(img_array)
    # print("Predicted:", decode_predictions(preds, top=1))

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    return heatmap


def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))


# save_and_display_gradcam(x[0], prepare_image(), cam_path=f"/content/drive/heatmap/covid")
