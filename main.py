import ImageConverter
import cv2
from SceneDetectionAudio import SceneDetectionAudio
from SceneDetectionVisual import SceneDetectionVisual
import ObjectDetectionVisual

if __name__ == '__main__':
    saliency_coarse = cv2.saliency.StaticSaliencySpectralResidual_create()
    scene_detection_visual = SceneDetectionVisual("places", "resnet18")
    scene_detection_audio = SceneDetectionAudio(
        "soundnet/sound8.npy",
        "/mnt/datadrive/projects/thesis/Datasets/Audio/Audio_Filtered",
        "soundnet/categories_imagenet.txt",
        "soundnet/categories_places2.txt",
        "soundnet/audio_object_detection_filtered.json",
        "soundnet/audio_scene_detection_filtered.json"
    )

    # scene_detection_audio.update_object_scene_detection_files()
    # scene_detection.detect_bulk("/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/")

    # ObjectDetectionVisual.detect_bulk("/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/")

    # ImageConverter.convert_paintings_to_txt_bulk("data/test/paintings", "data/test/paintings", True, True, True)

    # ImageConverter.convert_txt_to_sound_bulk("sound_engine.scd", "converted/")

    # ImageConverter.convert_paintings_to_txt_bulk("data/test/", "data/test/", True, True)

    # ImageConverter.convert_painting_to_presentation_bulk("data/test/", "data/presentation/", False, True, False, True)

    ImageConverter.convert_painting_to_presentation_bulk(
        "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/",
        "data/presentation_saliency/",
        saliency_coarse,
        True,
        (scene_detection_visual, scene_detection_audio),
        True,
        True,
        True,
        True,
        False)

    # ImageConverter.convert_painting_to_presentation_bulk(
    #     "data/test/paintings",
    #     "data/test/paintings",
    #     saliency_coarse,
    #     True,
    #     (scene_detection_visual, scene_detection_audio),
    #     True,
    #     True,
    #     False,
    #     True,
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
