# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch.autograd import Function

# pylint: disable=W0221
class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs,  torch.tensor(bound))
        return torch.clamp_min(inputs, bound)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, bound = ctx.saved_tensors
        pass_through_1 = inputs >= bound
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through * grad_output, None
