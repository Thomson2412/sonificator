import csv
import os
from pymongo import MongoClient


def main():
    with open("private/connection_string.txt", "r") as f:
        lines = f.readlines()
    if len(lines) == 1:
        client = MongoClient(lines[0])
        db = client.evaluator
        input_from_dir(
            "/mnt/datadrive/projects/thesis/sonificator/data/presentation_saliency/",
            db,
            "../data/combinations_exclusion.csv"
        )


def input_from_dir(input_dir, db, exclusion_csv):

    exclusion_dict = {}
    with open(exclusion_csv) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for key, value in row.items():
                if value != "":
                    if key in exclusion_dict:
                        exclusion_dict[key].append(value)
                    else:
                        exclusion_dict[key] = [value]

    image_filename_list = []
    video_filename_list = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if ".jpg" in filename or ".png" in filename:
                q = generate_questions(0, filename)
                input_question(q, db)
                image_filename_list.append(filename)
            if ".mp3" in filename:
                q = generate_questions(1, filename)
                input_question(q, db)
            if "_audio.mp4" in filename:
                video_filename_list.append(filename)

    image_pairs = [[i, j] for i in image_filename_list for j in image_filename_list if i != j]
    for image_filenames in image_pairs:
        is_same_category = False
        for values in exclusion_dict.values():
            if image_filenames[0] in values and image_filenames[1] in values:
                is_same_category = True
        if not is_same_category:
            audio_filename = f"{os.path.splitext(image_filenames[0])[0]}.mp3"
            image_filenames.append(audio_filename)
            q = generate_questions(3, image_filenames)
            input_question(q, db)


    # video_pairs = [[i, j] for i in video_filename_list for j in video_filename_list if i != j]
    # for video_filenames in video_pairs:
    #     q = generate_questions(2, video_filenames)
    #     input_question(q, db)


def generate_questions(question_type, filename):
    if question_type == 0:
        question_list = [
            {
                "type": 0,
                "question": "Would you state that the artwork contains a lot of different colors?",
                "scale": ["A few", "A lot"],
                "content": filename
            },
            {
                "type": 0,
                "question": "Would you describe the artwork as mostly pale or saturated?",
                "scale": ["Pale", "Saturated"],
                "content": filename
            },
            {
                "type": 0,
                "question": "Would you describe the artwork as mostly dark or light?",
                "scale": ["Light", "Dark"],
                "content": filename
            },
            {
                "type": 0,
                "question": "Would you describe the artwork as mostly smooth or rough?",
                "scale": ["Smooth", "Rough"],
                "content": filename
            }
        ]
    elif question_type == 1:
        question_list = [
            {
                "type": 1,
                "question": "Would you state that the musical piece contains a lot of different chords?",
                "scale": ["A few", "A lot"],
                "content": filename
            },
            {
                "type": 1,
                "question": "Would you describe the sound of the musical piece as mostly quiet or loud?",
                "scale": ["Quite", "Loud"],
                "content": filename
            },
            {
                "type": 1,
                "question": "Would you describe the notes of the musical piece as mostly low or high in pitch?",
                "scale": ["Low", "High"],
                "content": filename
            },
            {
                "type": 1,
                "question": "Would you describe the sound of the musical piece as mostly smooth or rough?",
                "scale": ["Smooth", "Rough"],
                "content": filename
            },
            {
                "type": 1,
                "question": "Would you describe the musical piece as pleasant?",
                "scale": ["Very unpleasant", "Very pleasant"],
                "content": filename
            }
        ]
    elif question_type == 2:
        question_list = [
            {
                "type": 2,
                "question": "Which painting is most fitting to the musical piece?",
                "scale": ["First", "Second"],
                "content": filename
            }
        ]
    elif question_type == 3:
        question_list = [
            {
                "type": 3,
                "question": "Which painting is most fitting to the musical piece?",
                "scale": ["First", "Second"],
                "content": filename
            }
        ]
    else:
        raise Exception
    return question_list


def input_question(data_list, db):
    result = db.questions.insert_many(
        data_list
    )
    print(result)


if __name__ == '__main__':
    main()
