import ImageConverter
import cv2
import SceneDetectionAudio
from SceneDetectionVisual import SceneDetectionVisual
import ObjectDetectionVisual

if __name__ == '__main__':
    # SceneDetectionAudio.update_object_scene_detection_files(
    #     "/mnt/datadrive/projects/thesis/Datasets/Audio/",
    #     "soundnet/",
    #     "soundnet/audio_object_detection.json",
    #     "soundnet/audio_scene_detection.json",
    # )


    # scene_detection = SceneDetectionVisual("resnet18")
    # scene_detection.detect_bulk("/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/")

    # ObjectDetectionVisual.detect_bulk("/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/")

    # ImageConverter.convert_paintings_to_txt_bulk("data/test/paintings", "data/test/paintings", True, True, True)

    # ImageConverter.convert_txt_to_sound_bulk("sound_engine.scd", "converted/")

    # ImageConverter.convert_paintings_to_txt_bulk("data/test/", "data/test/", True, True)

    # ImageConverter.convert_painting_to_presentation_bulk("data/test/", "data/presentation/", False, True, False, True)

    ImageConverter.convert_painting_to_presentation_bulk(
        "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/",
        "data/presentation_saliency/",
        True,
        True,
        True,
        True,
        True,
        True,
        False)

    # ImageConverter.convert_painting_to_presentation_bulk(
    #     "data/test/paintings",
    #     "data/test/paintings",
    #     True,
    #     True,
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
