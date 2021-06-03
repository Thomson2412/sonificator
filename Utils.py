import math

import numpy as np
from pythonosc import osc_message_builder


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


def create_message_from_list(address, data_list):
    msg = osc_message_builder.OscMessageBuilder(address=address)
    for data in data_list:
        if isinstance(data, list):
            msg.add_arg(bytes(data))
        else:
            msg.add_arg(data)
    return msg.build()


def calculate_step_priority(saliency_map, steps):
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
    return list({k: v for k, v in sorted(to_be_ordered.items(), key=lambda item: item[1])}.keys())


class Utils:
    pass
