import math
from collections import Counter
import numpy as np
from pythonosc import osc_message_builder
import scipy
import scipy.misc
import scipy.cluster


def scale_between_range(value, in_min_max, out_min_max):
    in_min = in_min_max[0]
    in_max = in_min_max[1]
    out_min = out_min_max[0]
    out_max = out_min_max[1]
    if in_min == in_max:
        return round((out_max - out_min) / 2)
    scaled = round(out_min + (value - in_min) * ((out_max + out_min) / (in_max - in_min)))
    if scaled > out_max:
        return round(out_max)
    if scaled < out_min:
        return round(out_min)
    return scaled


def scale_between_range_no_round(value, in_min_max, out_min_max):
    in_min = in_min_max[0]
    in_max = in_min_max[1]
    out_min = out_min_max[0]
    out_max = out_min_max[1]
    if in_min == in_max:
        return (out_max - out_min) / 2
    scaled = out_min + (value - in_min) * ((out_max + out_min) / (in_max - in_min))
    if scaled > out_max:
        return out_max
    if scaled < out_min:
        return out_min
    return scaled


def create_message_from_list(address, data_list):
    msg = osc_message_builder.OscMessageBuilder(address=address)
    for data in data_list:
        if isinstance(data, list):
            msg.add_arg(bytes(data))
        else:
            msg.add_arg(data)
    return msg.build()


def calculate_step_priority_standard(saliency_map, steps):
    sal_list = []
    width = int(saliency_map.shape[1])
    height = int(saliency_map.shape[0])
    step_size_x = math.ceil(width / math.sqrt(steps))
    step_size_y = math.ceil(height / math.sqrt(steps))
    for y in range(0, height, step_size_y):
        for x in range(0, width, step_size_x):
            sub_sal = saliency_map[y:y + step_size_y, x:x + step_size_x]
            sum_sub_sal = np.round(np.sum(sub_sal) / (sub_sal.shape[1] * sub_sal.shape[0])).astype(int)
            sal_list.append(sum_sub_sal)
    to_be_ordered = dict()
    for step in range(steps):
        to_be_ordered[step] = sal_list[step]
    return list({k: v for k, v in sorted(to_be_ordered.items(), reverse=True, key=lambda item: item[1])}.keys())


def calculate_step_priority_object(segmentation_img, thresh_map):
    to_be_ordered = dict()
    for mask_id in np.unique(segmentation_img):
        mask = segmentation_img == mask_id
        sal_segment = thresh_map[mask]
        sal_count = Counter(sal_segment)
        sal_percentage = round((sal_count[255] / (sal_count[0] + sal_count[255])) * 100)
        to_be_ordered[mask_id] = sal_percentage

    return list({k: v for k, v in sorted(to_be_ordered.items(), reverse=True, key=lambda item: item[1])}.keys())


def merge_segments_by_category(segmentation_img, segmentation_info):
    ids_for_category = {}
    for info in segmentation_info:
        category = info["category_id"]
        if category in ids_for_category.keys():
            ids_for_category[category].append(info["id"])
        else:
            ids_for_category[category] = [info["id"]]
    for key, id_values in ids_for_category.items():
        if len(id_values) > 1:
            min_id = min(id_values)
            for id_value in id_values:
                segmentation_img[segmentation_img == id_value] = min_id


def get_dominant_color(ar, num_clusters):
    ar = ar.astype(float)
    codes, _ = scipy.cluster.vq.kmeans2(ar, num_clusters)
    vecs, _ = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, _ = scipy.histogram(vecs, len(codes))  # count occurrences
    index_max = scipy.argmax(counts)  # find most frequent
    peak = codes[index_max]
    return peak.astype(np.uint8)


class Utils:
    pass
