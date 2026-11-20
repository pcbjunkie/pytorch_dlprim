"""Test the native SimOTA geometry predicate with OpenCL tensor operators."""
import os
from types import MethodType


def install(head):
    original = head.get_geometry_constraint

    def get_geometry_constraint(self, boxes, strides, x_shifts, y_shifts):
        if os.environ.get('PROFILE_X_GEOMETRY') != '1' or boxes.device.type != 'ocl':
            return original(boxes, strides, x_shifts, y_shifts)
        stride = strides[0]
        x = ((x_shifts[0] + 0.5) * stride).unsqueeze(0)
        y = ((y_shifts[0] + 0.5) * stride).unsqueeze(0)
        distance = stride.unsqueeze(0) * 1.5
        bx, by = boxes[:, 0:1], boxes[:, 1:2]
        inside = (x > bx - distance) & (x < bx + distance)
        inside = inside & (y > by - distance) & (y < by + distance)
        mask = inside.sum(dim=0) > 0
        return mask, inside[:, mask]

    head.get_geometry_constraint = MethodType(get_geometry_constraint, head)
