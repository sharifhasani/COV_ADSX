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
    return res


def load_DNN():
    pre_trained_model = DenseNet169(include_top=False, input_shape=(224, 224, 3), pooling="avg")
    return pre_trained_model

def load_XGBoost_model():
    model_path = join(MEDIA_ROOT, "MLModel", "xgb_model.pkl")
    xgb_model = pickle.load(open(model_path, "rb"))
    return xgb_model
