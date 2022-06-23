import ImageConverter
import cv2
from SceneDetectionAudio import SceneDetectionAudio
from SceneDetectionVisual import SceneDetectionVisual

if __name__ == '__main__':
    saliency_coarse = cv2.saliency.StaticSaliencySpectralResidual_create()
    scene_detection_visual = SceneDetectionVisual("places", "resnet18")
    scene_detection_audio = SceneDetectionAudio(
        "soundnet/sound8.npy",
        "datasets/scene_audio_dataset",
        "soundnet/categories_imagenet.txt",
        "soundnet/categories_places2.txt",
        "soundnet/audio_object_detection_scene_audio.json",
        "soundnet/audio_scene_detection_scene_audio.json",
        "datasets/scene_audio_dataset"
    )

    # scene_detection_audio.update_object_scene_detection_files()
    # scene_detection_audio.create_scene_audio_dataset_for_paintings(
    #     "datasets/painting_dataset", scene_detection_visual, True)
    # scene_detection_visual.detect_bulk(
    #     "datasets/painting_dataset")

    # Model 1
    ImageConverter.convert_painting_to_presentation_bulk(
        input_dir="datasets/painting_dataset",
        output_dir="output/model1/",
        sound_model="sound_engine_model1.scd",
        saliency_coarse=saliency_coarse,
        with_saliency=True,
        scene_detection=(scene_detection_visual, scene_detection_audio),
        use_object_nav=False,
        add_audio=True,
        web_convert=True,
        include_content=True,
        include_border=False,
        inner_scaling=False,
        things_as_chaos=False)

    # Model 2
    ImageConverter.convert_painting_to_presentation_bulk(
        input_dir="datasets/painting_dataset",
        output_dir="output/model2/",
        sound_model="sound_engine_model2.scd",
        saliency_coarse=saliency_coarse,
        with_saliency=True,
        scene_detection=(scene_detection_visual, scene_detection_audio),
        use_object_nav=True,
        add_audio=True,
        web_convert=True,
        include_content=True,
        include_border=False,
        inner_scaling=False,
        things_as_chaos=False)

    # Model 3
    ImageConverter.convert_painting_to_presentation_bulk(
        input_dir="datasets/painting_dataset",
        output_dir="output/model3/",
        sound_model="sound_engine_model3.scd",
        saliency_coarse=saliency_coarse,
        with_saliency=True,
        scene_detection=(scene_detection_visual, scene_detection_audio),
        use_object_nav=True,
        add_audio=True,
        web_convert=True,
        include_content=True,
        include_border=False,
        inner_scaling=False,
        things_as_chaos=False)

    # Model 4
    ImageConverter.convert_painting_to_presentation_bulk(
        input_dir="datasets/painting_dataset",
        output_dir="output/model4/",
        sound_model="sound_engine_model4.scd",
        saliency_coarse=saliency_coarse,
        with_saliency=True,
        scene_detection=(scene_detection_visual, scene_detection_audio),
        use_object_nav=True,
        add_audio=True,
        web_convert=True,
        include_content=True,
        include_border=False,
        inner_scaling=True,
        things_as_chaos=True)

    # Model 4 no scene
    # ImageConverter.convert_painting_to_presentation_bulk(
    #     input_dir="datasets/painting_dataset",
    #     output_dir="data/model4_no_scene/",
    #     sound_model="sound_engine_model4.scd",
    #     saliency_coarse=saliency_coarse,
    #     with_saliency=True,
    #     scene_detection=None,
    #     use_object_nav=True,
    #     add_audio=True,
    #     web_convert=True,
    #     include_content=True,
    #     include_border=False,
    #     inner_scaling=True,
    #     things_as_chaos=True)
