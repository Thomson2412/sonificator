import ImageConverter
import cv2
import SceneDetectionAudio

if __name__ == '__main__':
    # SceneDetectionAudio.update_object_scene_detection_files(
    #     "soundnet/mp3/",
    #     "soundnet",
    #     "data/test/audio_scene/object_audio_detection.json",
    #     "data/test/audio_scene/object_scene_detection.json",
    # )
    #
    # audio_paths = SceneDetectionAudio.get_audio_for_scene("data/test/audio_scene/object_scene_detection.json", "/r/railroad_track")
    # print(audio_paths)

    # ImageConverter.convert_paintings_to_txt_bulk("data/painter_by_numbers_scene_correct/", "converted/")

    # ImageConverter.convert_txt_to_sound_bulk("sound_engine.scd", "converted/")

    # ImageConverter.convert_paintings_to_txt_bulk("data/test/", "data/test/", True, True)

    # ImageConverter.convert_painting_to_presentation_bulk("data/test/", "data/presentation/", False, True, False, True)

    ImageConverter.convert_painting_to_presentation_bulk(
        "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/",
        "data/presentation_saliency/",
        True,
        False,
        True,
        True,
        True,
        False)

    # ImageConverter.convert_painting_to_presentation_bulk(
    #     "data/test",
    #     "data/test/",
    #     True,
    #     False,
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
