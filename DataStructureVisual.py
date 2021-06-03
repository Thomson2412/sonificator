from collections import OrderedDict

import cv2
import numpy as np


class DataStructureVisual:
    def __init__(self, original_img, hsv_img, edge_img, overall_dominant_img, saliency_heatmap_img,
                 saliency_thresh_map_img, steps):
        self.original_img = original_img
        self.hsv_img = hsv_img
        self.edge_img = edge_img
        self.overall_dominant_img = overall_dominant_img
        self.saliency_heatmap_img = saliency_heatmap_img
        self.saliency_thresh_map_img = saliency_thresh_map_img

        self.steps = steps

        self.sub_presentation_img = OrderedDict()

        self.append_count = 0
        self.seen_priorities = []

    def append_sub_img(self, sub_presentation_img, x, y, duration, priority):
        if priority in self.seen_priorities:
            raise KeyError("Duplicate priority not allowed")
        self.seen_priorities.append(priority)
        self.sub_presentation_img[priority] = {
            "content": sub_presentation_img,
            "x": x,
            "y": y,
            "duration": duration
        }
        self.append_count += 1

    def get_presentation_for_step(self, step, include_content):
        self.assert_condition()
        step += 1
        if step > self.steps:
            raise AssertionError("Step higher than available")
        if step < 0:
            raise AssertionError("Step can't be below zero")

        presentation = np.array(self.original_img, copy=True)
        for i in range(step):
            partial = self.sub_presentation_img[i]
            x = partial["x"]
            y = partial["y"]
            content = partial["content"]
            presentation[y:y + content.shape[0], x:x + content.shape[1]] = self.get_segment_for_step(i, include_content)
        return presentation

    def get_segment_for_step(self, step, include_content):
        self.assert_condition()
        if step >= self.steps:
            raise AssertionError("Step higher than available")
        if step < 0:
            raise AssertionError("Step can't be below zero")

        partial = self.sub_presentation_img[step]
        x = partial["x"]
        y = partial["y"]
        content = partial["content"]
        if include_content:
            return content
        else:
            border_size = 5
            border_color = (0, 255, 0)
            presentation = np.array(self.original_img, copy=True)
            presentation = presentation[y:y + content.shape[0], x:x + content.shape[1]]
            return cv2.rectangle(presentation, (0, 0), (content.shape[1], content.shape[0]), border_color, border_size)

    def generate_presentation_video(self, output_file, include_content):
        self.assert_condition()
        fps = 1
        size = (self.original_img.shape[1], self.original_img.shape[0])
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for step in range(self.steps):
            partial = self.sub_presentation_img[step]
            duration = partial["duration"]
            img = self.get_presentation_for_step(step, include_content)
            for i in range(duration * fps):
                out.write(img)
        out.release()

    def assert_condition(self):
        if self.append_count != self.steps:
            raise AssertionError("append_count inconsistent")
        if len(self.sub_presentation_img) != self.steps:
            raise AssertionError("hue inconsistent")
