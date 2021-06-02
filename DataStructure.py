import array
from collections import OrderedDict


def dic_to_string(input_dic, sorted_by_key, line_ending=0, delimiter=" "):
    if sorted_by_key:
        keys = sorted(input_dic.keys())
    else:
        keys = input_dic.keys()

    result = str()
    for key in keys:
        value = input_dic[key]
        if isinstance(value, (int, float, complex)):
            result += f"{input_dic[key]}{delimiter}"

        elif isinstance(value, list):
            result += f"{delimiter.join(str(v) for v in value)}\r\n"

    for i in range(line_ending):
        result += "\r\n"
    return result


class DataStructure:
    def __init__(self, root, scale, steps):
        self.root = root
        self.scale = scale
        self.steps = steps
        self.hue = OrderedDict()
        self.saturation = OrderedDict()
        self.intensity = OrderedDict()
        self.duration = OrderedDict()
        self.edginess = OrderedDict()
        self.line = OrderedDict()
        self.append_count = 0
        self.seen_priorities = []

    def append_sub_img(self, hue, saturation, intensity, duration, edginess, line, priority):
        if priority in self.seen_priorities:
            raise KeyError("Duplicate priority not allowed")
        self.seen_priorities.append(priority)
        self.hue[priority] = hue
        self.saturation[priority] = saturation
        self.intensity[priority] = intensity
        self.duration[priority] = duration
        self.edginess[priority] = edginess
        self.line[priority] = line
        self.append_count += 1

    def write_to_file(self, output_file, with_priority_order):
        if self.append_count != self.steps:
            raise AssertionError("append_count inconsistent")
        if len(self.hue) != self.steps:
            raise AssertionError("hue inconsistent")
        if len(self.saturation) != self.steps:
            raise AssertionError("saturation inconsistent")
        if len(self.intensity) != self.steps:
            raise AssertionError("intensity inconsistent")
        if len(self.duration) != self.steps:
            raise AssertionError("duration inconsistent")
        if len(self.edginess) != self.steps:
            raise AssertionError("edginess inconsistent")
        if len(self.line) != self.steps:
            raise AssertionError("line inconsistent")

        with open(output_file, mode="w") as file:
            file.write(f"{self.root}\r\n\r\n")

            file.write(f"{self.scale}\r\n\r\n")

            file.write(f"{self.steps}\r\n\r\n")

            file.write(dic_to_string(self.hue, with_priority_order, 2))

            file.write(dic_to_string(self.saturation, with_priority_order, 2))

            file.write(dic_to_string(self.intensity, with_priority_order, 2))

            file.write(dic_to_string(self.duration, with_priority_order, 2))

            file.write(dic_to_string(self.edginess, with_priority_order, 2))

            file.write(dic_to_string(self.line, with_priority_order, 1))

            if with_priority_order:
                file.write(' '.join(str(p) for p in self.seen_priorities))
            else:
                file.write(' '.join(str(p) for p in sorted(self.seen_priorities)))

