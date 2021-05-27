class DataStructure:
    def __init__(self, root, scale, steps):
        self.root = root
        self.scale = scale
        self.steps = steps
        self.hue = []
        self.saturation = []
        self.intensity = []
        self.duration = []
        self.edginess = []
        self.line = []

    def append_sub_img(self, hue, saturation, intensity, duration, edginess, line):
        self.hue.append(hue)
        self.saturation.append(saturation)
        self.intensity.append(intensity)
        self.duration.append(duration)
        self.edginess.append(edginess)
        self.line.append(line)

    def write_to_file(self, output_file):
        with open(output_file, mode="w") as file:
            file.write(str(self.root))
            file.write("\r\n")
            file.write("\r\n")

            file.write(str(self.scale))
            file.write("\r\n")
            file.write("\r\n")

            file.write(str(self.steps))
            file.write("\r\n")
            file.write("\r\n")

            file.write(" ".join(str(x) for x in self.hue))
            file.write("\r\n")
            file.write("\r\n")

            file.write(" ".join(str(x) for x in self.saturation))
            file.write("\r\n")
            file.write("\r\n")

            file.write(" ".join(str(x) for x in self.intensity))
            file.write("\r\n")
            file.write("\r\n")

            file.write(" ".join(str(x) for x in self.duration))
            file.write("\r\n")
            file.write("\r\n")

            file.write(" ".join(str(x) for x in self.edginess))
            file.write("\r\n")
            file.write("\r\n")

            for line in self.line:
                file.write(" ".join(str(x) for x in line))
                file.write("\r\n")
