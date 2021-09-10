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
    #     True)

    # ImageConverter.convert_painting_to_presentation_bulk("data/test/", "data/presentation/", False, True, False, True)

    ImageConverter.convert_painting_to_presentation_bulk(
        "/mnt/datadrive/projects/thesis/Datasets/Paintings/evaluation_dataset/",
        "data/presentation_scene/",
        "sound_engine_model3.scd",
        saliency_coarse,
        True,
        (scene_detection_visual, scene_detection_audio),
        True,
        True,
        False,
        True,
        False,
        True)

    # ImageConverter.convert_painting_to_presentation_bulk(
    #     "data/test/paintings",
    #     "data/test/paintings",
    #     "sound_engine_model3.scd",
    #     saliency_coarse,
    #     True,
    #     (scene_detection_visual, scene_detection_audio),
    #     True,
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
