from collections import OrderedDict


def dic_to_string(input_dic, line_ending=0, delimiter=" "):
    result = str()
    for i, key in enumerate(sorted(input_dic.keys())):
        value = input_dic[key]
        if isinstance(value, (int, float, complex)):
            if len(input_dic) - 1 == i:
                result += f"{input_dic[key]}"
            else:
                result += f"{input_dic[key]}{delimiter}"

        elif isinstance(value, list):
            result += f"{delimiter.join(str(v) for v in value)}\r\n"

    for i in range(line_ending):
        result += "\r\n"
    return result


class DataStructureAudio:
    def __init__(self, root, scale, steps):
        self.root = root
        self.scale = scale

        self.steps = steps

        self.hue = OrderedDict()
        self.saturation = OrderedDict()
        self.intensity = OrderedDict()
        self.pan = OrderedDict()
        self.duration = OrderedDict()
        self.edginess = OrderedDict()
        self.line = OrderedDict()

        self.append_count = 0
        self.seen_priorities = []

    def append_sub_img(self, hue, saturation, intensity, pan, duration, edginess, line, priority):
        if priority in self.seen_priorities:
            raise KeyError("Duplicate priority not allowed")
        self.seen_priorities.append(priority)
        self.hue[priority] = hue
        self.saturation[priority] = saturation
        self.intensity[priority] = intensity
        self.pan[priority] = pan
        self.duration[priority] = duration
        self.edginess[priority] = edginess
        self.line[priority] = line
        self.append_count += 1

    def write_to_file(self, output_file):
        self.assert_condition()
        with open(output_file, mode="w") as file:
            file.write(f"{self.root}\r\n\r\n")

            file.write(f"{self.scale}\r\n\r\n")

            file.write(f"{self.steps}\r\n\r\n")

            file.write(dic_to_string(self.hue, 2))

            file.write(dic_to_string(self.saturation, 2))

            file.write(dic_to_string(self.intensity, 2))

            file.write(dic_to_string(self.pan, 2))

            file.write(dic_to_string(self.duration, 2))

            file.write(dic_to_string(self.edginess, 2))

            file.write(dic_to_string(self.line, 1))

            file.write(' '.join(str(p) for p in self.seen_priorities))

    def assert_condition(self):
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
