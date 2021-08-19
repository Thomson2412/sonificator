import os
from shutil import copyfile
from tensorflow.keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, Input
from tensorflow.keras.models import Model
import numpy as np
import librosa
import json

from SceneDetectionVisual import SceneDetectionVisual


def preprocess(audio):
    audio *= 256.0  # SoundNet needs the range to be between -256 and 256
    # reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    return audio


def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on mono audio files with a sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=sample_rate, mono=True)
    audio = preprocess(audio)
    return audio


def predictions_to_categories(prediction, cat):
    scenes = []
    for p in range(prediction.shape[1]):
        index = np.argmax(prediction[0, p, :])
        scenes.append(cat[index])
    return scenes


class SceneDetectionAudio:
    def __init__(self, model_path, dataset_base_dir, object_categories, scene_categories,
                 object_data_file, scene_data_file):
        self.model_path = model_path
        self.dataset_base_dir = dataset_base_dir
        self.object_categories = object_categories
        self.scene_categories = scene_categories
        self.object_data_file = object_data_file
        self.scene_data_file = scene_data_file

    def build_model(self):
        model_weights = np.load(self.model_path, encoding="latin1", allow_pickle=True).item()
        model_weights = dict(model_weights)
        seq_layer_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                                 'kernel_size': 64, 'conv_strides': 2,
                                 'pool_size': 8, 'pool_strides': 8},

                                {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                                 'kernel_size': 32, 'conv_strides': 2,
                                 'pool_size': 8, 'pool_strides': 8},

                                {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                                 'kernel_size': 16, 'conv_strides': 2},

                                {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                                 'kernel_size': 8, 'conv_strides': 2},

                                {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                                 'kernel_size': 4, 'conv_strides': 2,
                                 'pool_size': 4, 'pool_strides': 4},

                                {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                                 'kernel_size': 4, 'conv_strides': 2},

                                {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                                 'kernel_size': 4, 'conv_strides': 2}
                                ]

        # Build sequential layers
        inputs = Input(shape=(None, 1))
        prev_layer = inputs
        for seq_layer_parameter in seq_layer_parameters:
            prev_layer = ZeroPadding1D(padding=seq_layer_parameter['padding'])(prev_layer)

            conv_layer = Conv1D(seq_layer_parameter['num_filters'],
                                kernel_size=seq_layer_parameter['kernel_size'],
                                strides=seq_layer_parameter['conv_strides'],
                                padding='valid',
                                name=seq_layer_parameter['name'])

            prev_layer = conv_layer(prev_layer)
            weights = model_weights[seq_layer_parameter['name']]['weights'].reshape(conv_layer.get_weights()[0].shape)
            biases = model_weights[seq_layer_parameter['name']]['biases']
            conv_layer.set_weights([weights, biases])

            gamma = model_weights[seq_layer_parameter['name']]['gamma']
            beta = model_weights[seq_layer_parameter['name']]['beta']
            mean = model_weights[seq_layer_parameter['name']]['mean']
            var = model_weights[seq_layer_parameter['name']]['var']
            batch_norm = BatchNormalization()
            prev_layer = batch_norm(prev_layer)
            batch_norm.set_weights([gamma, beta, mean, var])

            prev_layer = Activation('relu')(prev_layer)

            if 'pool_size' in seq_layer_parameter:
                prev_layer = MaxPooling1D(pool_size=seq_layer_parameter['pool_size'],
                                          strides=seq_layer_parameter['pool_strides'],
                                          padding='valid')(prev_layer)

        # Build split output layers
        object_output_layer = ZeroPadding1D(padding=0)(prev_layer)
        conv_layer = Conv1D(1000, kernel_size=8, strides=2, padding='valid', name='conv8')
        object_output_layer = conv_layer(object_output_layer)
        weights = model_weights['conv8']['weights'].reshape(conv_layer.get_weights()[0].shape)
        biases = model_weights['conv8']['biases']
        conv_layer.set_weights([weights, biases])

        scene_output_layer = ZeroPadding1D(padding=0)(prev_layer)
        conv_layer = Conv1D(401, kernel_size=8, strides=2, padding='valid', name='conv8_2')
        scene_output_layer = conv_layer(scene_output_layer)
        weights = model_weights['conv8_2']['weights'].reshape(conv_layer.get_weights()[0].shape)
        biases = model_weights['conv8_2']['biases']
        conv_layer.set_weights([weights, biases])

        return Model(inputs=inputs, outputs=[object_output_layer, scene_output_layer])

    def update_object_scene_detection_files(self):
        model_object = self.build_model()
        model_object.summary()
        with open(self.object_categories, 'r') as f:
            object_categories = f.read().split('\n')
        with open(self.scene_categories, 'r') as f:
            scene_categories = f.read().split('\n')

        for root, dirs, files in os.walk(self.dataset_base_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                key_name = os.path.relpath(file_path, self.dataset_base_dir)
                prediction_result_object = {}
                prediction_result_scene = {}
                if os.path.isfile(self.object_data_file):
                    with open(self.object_data_file) as json_file:
                        prediction_result_object = json.load(json_file)
                if os.path.isfile(self.scene_data_file):
                    with open(self.scene_data_file) as json_file:
                        prediction_result_scene = json.load(json_file)
                if ".wav" in filename and (key_name not in prediction_result_object.keys()
                                           or key_name not in prediction_result_scene.keys()):
                    duration = librosa.get_duration(filename=file_path)
                    if duration >= 5.1:
                        print(duration)
                        try:
                            plain_prediction = model_object.predict(load_audio(file_path))
                            object_pred_cat = predictions_to_categories(plain_prediction[0], object_categories)
                            scene_pred_cat = predictions_to_categories(plain_prediction[1], scene_categories)

                            prediction_result_object[key_name] = {
                                "audio_length": duration,
                                "prediction": object_pred_cat
                            }
                            prediction_result_scene[key_name] = {
                                "audio_length": duration,
                                "prediction": scene_pred_cat
                            }

                            with open(self.object_data_file, "w") as prediction_result_object_outfile:
                                json.dump(prediction_result_object, prediction_result_object_outfile, indent=4)
                            with open(self.scene_data_file, "w") as prediction_result_scene_outfile:
                                json.dump(prediction_result_scene, prediction_result_scene_outfile, indent=4)
                        except:
                            print("Something went wrong")

    def get_audio_for_scene_json(self, scene):
        scene_results = []
        prediction_result_scene = {}
        if os.path.isfile(self.scene_data_file):
            with open(self.scene_data_file) as json_file:
                prediction_result_scene = json.load(json_file)
        for audio, values in prediction_result_scene.items():
            for pred in values["prediction"]:
                if scene in pred:
                    scene_results.append(os.path.join(self.dataset_base_dir, audio))
        return scene_results

    def get_audio_for_scene_folder(self, audio_folder, scene):
        scene_results = []
        for root, dirs, files in os.walk(audio_folder):
            for filename in files:
                if root.endswith(scene.replace("/", "_")) and ".wav" in filename:
                    scene_results.append(os.path.abspath(os.path.join(root, filename)))
        return scene_results

    def create_scene_audio_dataset_for_paintings(self, input_folder_img, output_dir, structure_only):
        scene_detection = SceneDetectionVisual("../places", "resnet18")
        for root, dirs, files in os.walk(input_folder_img):
            for filename in files:
                print(f"Working on: {filename}")
                file_path = os.path.join(root, filename)
                scene = scene_detection.detect(file_path).replace("/", "_")
                output_path = os.path.join(output_dir, scene)
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                    if not structure_only:
                        scene_audio_paths = self.get_audio_for_scene_json(scene)
                        for audio_path in scene_audio_paths:
                            filename_audio = os.path.basename(audio_path)
                            output_file_path = os.path.join(output_path, filename_audio)
                            copyfile(audio_path, output_file_path)
                print(f"Done: {filename}")

    def get_audio_for_object(self, object_name):
        object_results = []
        prediction_result_object = {}
        if os.path.isfile(self.object_data_file):
            with open(self.object_data_file) as json_file:
                prediction_result_object = json.load(json_file)
        for audio, values in prediction_result_object.items():
            for pred in values["prediction"]:
                if object_name in pred:
                    object_results.append(os.path.join(self.dataset_base_dir, audio))
        return object_results
