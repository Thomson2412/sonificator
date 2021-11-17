import os.path

import ImageConverter
import cv2
from SceneDetectionAudio import SceneDetectionAudio
from SceneDetectionVisual import SceneDetectionVisual

if __name__ == '__main__':
    saliency_coarse = cv2.saliency.StaticSaliencySpectralResidual_create()
    scene_detection_visual = SceneDetectionVisual("places", "resnet18")
    scene_detection_audio = SceneDetectionAudio(
        "soundnet/sound8.npy",
        "data/scene_audio",
        "soundnet/categories_imagenet.txt",
        "soundnet/categories_places2.txt",
        "soundnet/audio_object_detection_scene_audio.json",
        "soundnet/audio_scene_detection_scene_audio.json",
        "/mnt/datadrive/projects/thesis/Datasets/Audio/new_dataset_audio"
    )

    # scene_detection_audio.update_object_scene_detection_files()
    # scene_detection_audio.create_scene_audio_dataset_for_paintings(
    #     "/mnt/datadrive/projects/thesis/Datasets/Paintings/new_dataset_scene_correct", scene_detection_visual, True)
    # scene_detection_visual.detect_bulk(
    #     "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/")

    # ImageConverter.convert_paintings_to_txt_bulk(
    #     "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/",
    #     "data/presentation_scene/", saliency_coarse, True,
    #     (scene_detection_visual, scene_detection_audio), True)

    # ImageConverter.convert_txt_to_sound_bulk("sound_engine_model1.scd", "converted/")

    # ImageConverter.convert_paintings_to_txt_bulk(
    #     "data/test/paintings",
    #     "data/test/paintings",
    #     saliency_coarse,
    #     True,
    #     (scene_detection_visual, scene_detection_audio),
    #     True,
    #     False)

    # ImageConverter.convert_painting_to_presentation_bulk("data/test/", "data/presentation/", False, True, False, True)

    # ImageConverter.convert_painting_to_presentation_bulk(
    #     input_dir="/mnt/datadrive/projects/thesis/Datasets/Paintings/evaluation_dataset/",
    #     output_dir="data/presentation_evaluation_model_4/",
    #     sound_model=f"sound_engine_model4.scd",
    #     saliency_coarse=saliency_coarse,
    #     with_saliency=True,
    #     scene_detection=(scene_detection_visual, scene_detection_audio),
    #     use_object_nav=True,
    #     add_audio=True,
    #     web_convert=False,
    #     include_content=True,
    #     include_border=False,
    #     inner_scaling=True)

    # for i in range(1, 5):
    #     output_dir = f"data/presentation_evaluation_model_{i}_no_scene/"
    #     if not os.path.exists(output_dir):
    #         os.mkdir(output_dir)
    #     ImageConverter.convert_painting_to_presentation_bulk(
    #         input_dir="/mnt/datadrive/projects/thesis/Datasets/Paintings/evaluation_dataset/",
    #         output_dir=output_dir,
    #         sound_model=f"sound_engine_model{i}.scd",
    #         saliency_coarse=saliency_coarse,
    #         with_saliency=True,
    #         scene_detection=None,
    #         use_object_nav=True,
    #         add_audio=True,
    #         web_convert=True,
    #         include_content=True,
    #         include_border=False,
    #         inner_scaling=True)

    ImageConverter.convert_painting_to_presentation_bulk(
        input_dir="/mnt/datadrive/projects/thesis/Datasets/Paintings/evaluation_dataset/",
        output_dir="data/visual_model_4_sound_model_4",
        sound_model=f"sound_engine_model4.scd",
        saliency_coarse=saliency_coarse,
        with_saliency=True,
        scene_detection=None,
        use_object_nav=True,
        add_audio=True,
        web_convert=True,
        include_content=True,
        include_border=False,
        inner_scaling=True)

    # ImageConverter.convert_painting_to_presentation_bulk(
    #     "data/test/paintings",
    #     "data/test/paintings",
    #     "sound_engine_model4.scd",
    #     saliency_coarse,
    #     True,
    #     None,
    #     False,
    #     True,
    #     False,
    #     True,
    #     False,
    #     False)

    # saliency_coarse = cv2.saliency.StaticSaliencySpectralResidual_create()
    # ImageConverter.convert_painting_to_presentation(
    #     "data/test/allegorical_painting-Arthur_Streeton-Spirit_of_the_drought-1895-65154.jpg",
    #     "data/test/",
    #     saliency_coarse,
    #     True,
    #     None,
    #     False,
    #     True,
    #     True,
    #     True,
    #     False)
