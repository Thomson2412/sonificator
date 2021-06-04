import ImageConverter


if __name__ == '__main__':
    # ImageConverter.convert_paintings_to_txt_bulk("data/painter_by_numbers_scene_correct/", "converted/")

    # ImageConverter.convert_txt_to_sound_bulk("sound_engine.scd", "converted/")

    # ImageConverter.convert_paintings_to_txt_bulk("data/test/", "data/test/", False)

    # ImageConverter.convert_painting_to_presentation_bulk("data/test/", "data/presentation/", False, True, False, True)

    ImageConverter.convert_painting_to_presentation_bulk(
        "data/test/",
        "data/presentation_saliency/",
        True,
        True,
        True,
        False)

    # ImageConverter.add_audio_to_video(
    #     "data/presentation/painting10.avi",
    #     "data/presentation/painting10.aiff",
    #     "data/presentation/output.avi")
