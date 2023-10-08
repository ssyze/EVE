# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import copy


class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()

        self.experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output