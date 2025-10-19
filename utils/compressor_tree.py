import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from collections import deque
from typing import Tuple, List, Dict, Any, Callable
import logging
import time
import json
import torch
import torch.utils
import torch.utils.tensorboard
from abc import ABC, abstractmethod


from . import adder_utils
from .common import lse_gamma

try:
    import pulp
except ImportError:
    print(
        "pulp is not installed. Please install it using 'pip install pulp' to use MIP methods."
    )


def get_target_delay(bit_width):
    if bit_width == 8:
        return [50, 250, 400, 650]
    elif bit_width == 16:
        return [50, 200, 500, 1200]
    elif bit_width == 32:
        return [50, 300, 600, 2000]
    else:
        return [50, 600, 1500, 3000]


def get_full_target_delay(bit_width):
    if bit_width == 8:
        return list(range(50, 1000, 10))
    elif bit_width == 16:
        return list(range(50, 2000, 20))
    elif bit_width == 32:
        return list(range(50, 3000, 20))
    else:
        return list(range(50, 4000, 20))


def get_initial_partial_product(bit_width: int, encode_type: str) -> np.ndarray:
    if encode_type == "and":
        pp = np.zeros([bit_width * 2 - 1])
        for i in range(0, bit_width):
            pp[i] = i + 1
        for i in range(bit_width, bit_width * 2 - 1):
            pp[i] = bit_width * 2 - 1 - i
    elif encode_type == "booth":
        pp = np.zeros([bit_width * 2])
        for column_index in range(bit_width):
            if column_index % 2 == 0:
                pp[column_index] = 2 + column_index // 2
            else:
                pp[column_index] = 1 + column_index // 2
        pp[bit_width] = 1 + bit_width // 2
        pp[bit_width + 1] = 1 + bit_width // 2
        pp[bit_width + 2] = 1 + bit_width // 2
        pp[bit_width + 3] = 1 + bit_width // 2
        for column_index in range(bit_width + 4, bit_width * 2):
            if column_index % 2 == 0:
                pp[column_index] = bit_width // 2 - (column_index - bit_width - 4) // 2
            else:
                pp[column_index] = (
                    bit_width // 2 - (column_index - bit_width - 4) // 2 - 1
                )
    elif encode_type == "and_mac":
        pp = np.zeros([bit_width * 2 - 1])
        for i in range(0, bit_width):
            pp[i] = i + 1
        for i in range(bit_width, bit_width * 2 - 1):
            pp[i] = bit_width * 2 - 1 - i
        for i in range(0, bit_width):
            pp[i] += 1
    elif encode_type == "booth_mac":
        pp = np.zeros([bit_width * 2])
        for column_index in range(bit_width):
            if column_index % 2 == 0:
                pp[column_index] = 2 + column_index // 2
            else:
                pp[column_index] = 1 + column_index // 2
        pp[bit_width] = 1 + bit_width // 2
        pp[bit_width + 1] = 1 + bit_width // 2
        pp[bit_width + 2] = 1 + bit_width // 2
        pp[bit_width + 3] = 1 + bit_width // 2
        for column_index in range(bit_width + 4, bit_width * 2):
            if column_index % 2 == 0:
                pp[column_index] = bit_width // 2 - (column_index - bit_width - 4) // 2
            else:
                pp[column_index] = (
                    bit_width // 2 - (column_index - bit_width - 4) // 2 - 1
                )
        for i in range(0, bit_width):
            pp[i] += 1
    else:
        raise NotImplementedError

    return pp.astype(int)


class CompressorTree(ABC):
    def __init__(self, pp=None, ct32=None, ct22=None):
        self.pp: np.ndarray = pp
        self.ct32: np.ndarray = ct32
        self.ct22: np.ndarray = ct22

    @classmethod
    def from_dict(cls, d):
        ct32 = np.asarray(d["ct32"]).astype(int)
        ct22 = np.asarray(d["ct22"]).astype(int)
        pp = np.asarray(d["pp"]).astype(int)
        return cls(pp, ct32, ct22)

    @classmethod
    def wallace(cls, pp):
        max_stage_num = len(pp)
        stage_num = 0

        sequence_pp = np.zeros([1, len(pp)])
        sequence_pp[0] = copy.deepcopy(pp)
        ct32 = np.zeros([1, len(pp)])
        ct22 = np.zeros([1, len(pp)])
        target = np.asarray([2 for i in range(len(pp))])

        while stage_num < max_stage_num:
            for i in range(0, len(pp)):
                if sequence_pp[stage_num][i] % 3 == 0:
                    ct32[stage_num][i] = sequence_pp[stage_num][i] // 3
                    ct22[stage_num][i] = 0
                elif sequence_pp[stage_num][i] % 3 == 1:
                    ct32[stage_num][i] = sequence_pp[stage_num][i] // 3
                    ct22[stage_num][i] = 0
                elif sequence_pp[stage_num][i] % 3 == 2:
                    ct32[stage_num][i] = sequence_pp[stage_num][i] // 3
                    if stage_num == 0:
                        ct22[stage_num][i] = 0
                    else:
                        ct22[stage_num][i] = 1
            sequence_pp = np.r_[sequence_pp, np.zeros([1, len(pp)])]
            sequence_pp[stage_num + 1][0] = (
                sequence_pp[stage_num][0] - ct32[stage_num][0] * 2 - ct22[stage_num][0]
            )
            for i in range(1, len(pp)):
                sequence_pp[stage_num + 1][i] = (
                    sequence_pp[stage_num][i]
                    + ct32[stage_num][i - 1]
                    + ct22[stage_num][i - 1]
                    - ct32[stage_num][i] * 2
                    - ct22[stage_num][i]
                )
            stage_num += 1
            if (sequence_pp[stage_num] <= target).all():
                break

            ct32 = np.r_[ct32, np.zeros([1, len(pp)])]
            ct22 = np.r_[ct22, np.zeros([1, len(pp)])]

        assert (
            stage_num < max_stage_num
        ), "Exceed max stage num! Set max_stage_num larger"
        ct32 = np.sum(ct32, axis=0).astype(int)
        ct22 = np.sum(ct22, axis=0).astype(int)

        compressor_tree = cls(pp, ct32, ct22)
        return compressor_tree

    @classmethod
    def dadda(cls, pp):
        max_stage_num = len(pp)
        d = []
        d_j = 2
        for j in range(max_stage_num):
            d.append(d_j)
            d_j = int(np.floor(1.5 * d_j))

        remain_pp = copy.deepcopy(pp)

        ct32 = np.zeros(len(pp)).astype(int)
        ct22 = np.zeros(len(pp)).astype(int)

        for j in range(max_stage_num - 1, 0 - 1, -1):
            d_j = d[j]
            i = 0

            while i <= len(remain_pp) - 1:
                if remain_pp[i] <= d_j:
                    i += 1
                    continue
                elif remain_pp[i] == d_j + 1:
                    ct22[i] += 1
                    if i + 1 < len(remain_pp):
                        remain_pp[i + 1] += 1
                    remain_pp[i] -= 1
                    i += 1
                    continue
                else:
                    ct32[i] += 1
                    if i + 1 < len(remain_pp):
                        remain_pp[i + 1] += 1
                    remain_pp[i] -= 2
                    continue
        compressor_tree = cls(pp, ct32, ct22)
        return compressor_tree

    @classmethod
    def gomil(
        cls,
        pp,
        max_stage_num,
        method,
        alpha,
        beta,
        timeLimit,
        n_processing,
    ):
        raise NotImplementedError("More code will be released once accepted")

    @classmethod
    def ufomac(cls, pp):
        raise NotImplementedError("More code will be released once accepted")

    def to_dict(self):
        return {
            "ct32": self.ct32.tolist(),
            "ct22": self.ct22.tolist(),
        }

    def compressor_assignment(self):
        assert len(self.pp) == len(self.ct32) and len(self.pp) == len(self.ct22)
        column_len = len(self.pp)
        ct32_remain = copy.deepcopy(self.ct32)
        ct22_remain = copy.deepcopy(self.ct22)
        current_stage = 0
        ct32_decomposed = np.zeros([1, len(self.ct32)]).astype(int)
        ct22_decomposed = np.zeros([1, len(self.ct22)]).astype(int)

        current_pp = copy.deepcopy(self.pp)
        while (ct32_remain > 0).any() or (ct22_remain > 0).any():
            next_pp = np.zeros_like(current_pp)
            for column_index in range(column_len):
                current_pp_height = int(current_pp[column_index])
                assigned_ct32 = min(
                    int(ct32_remain[column_index]),
                    current_pp_height // 3,
                )
                ct32_remain[column_index] -= assigned_ct32
                current_pp_height -= assigned_ct32 * 3
                assigned_ct22 = min(
                    int(ct22_remain[column_index]),
                    current_pp_height // 2,
                )
                ct22_remain[column_index] -= assigned_ct22
                current_pp_height -= assigned_ct22 * 2
                next_pp[column_index] += (
                    current_pp_height + assigned_ct32 + assigned_ct22
                )
                if column_index < column_len - 1:
                    next_pp[column_index + 1] += assigned_ct32 + assigned_ct22

                ct32_decomposed[current_stage][column_index] = assigned_ct32
                ct22_decomposed[current_stage][column_index] = assigned_ct22
            current_pp = next_pp
            ct32_decomposed = np.r_[ct32_decomposed, np.zeros([1, len(self.ct32)])]
            ct22_decomposed = np.r_[ct22_decomposed, np.zeros([1, len(self.ct22)])]
            current_stage += 1
        ct32_decomposed = ct32_decomposed.astype(int)
        ct22_decomposed = ct22_decomposed.astype(int)
        if np.sum(ct32_decomposed[-1, :]) == 0 and np.sum(ct22_decomposed[-1, :]) == 0:
            ct32_decomposed = ct32_decomposed[:-1]
            ct22_decomposed = ct22_decomposed[:-1]
        return ct32_decomposed, ct22_decomposed

    def compressor_assignment_fused(self):
        dec_ct32, dec_ct22 = self.compressor_assignment()
        assignment = []
        stage_num, col_num = dec_ct32.shape
        ct32_counter = np.zeros_like(self.pp, dtype=int)
        ct22_counter = np.zeros_like(self.pp, dtype=int)

        for s in range(stage_num):
            stage_assignment = []
            for c in range(col_num):
                column_assignment = []
                for ct32_idx in range(dec_ct32[s, c]):
                    compressor_info = (s, c, 0, ct32_counter[c])
                    ct32_counter[c] += 1
                    column_assignment.append(compressor_info)
                for ct22_idx in range(dec_ct22[s, c]):
                    compressor_info = (s, c, 1, ct22_counter[c])
                    ct22_counter[c] += 1
                    column_assignment.append(compressor_info)
                stage_assignment.append(column_assignment)
            assignment.append(stage_assignment)
        assert (ct32_counter == self.ct32).all() and (ct22_counter == self.ct22).all()
        return assignment

    def compressor_assignment_ufomac(
        self,
        max_stage_num,
        M,
        method,
        n_processing,
        timeLimit,
        keepFiles,
    ):
        raise NotImplementedError("More code will be released once accepted")

    UFO_MAC_CONSTANT = {
        "FA": {
            "Tas": None,
            "Tac": None,
            "Tbs": None,
            "Tbc": None,
            "Tcs": None,
            "Tcc": None,
        },
        "HA": {
            "Tas": None,
            "Tac": None,
            "Tcs": None,
            "Tcc": None,
        },
    }

    def ufomac_router(
        self,
        dec_ct32: np.ndarray,
        dec_ct22: np.ndarray,
        method,
        Z_constant,
        n_processing,
        timeLimit,
        keepFiles,
        json_file,
        **kwargs,
    ) -> Tuple[str, List[deque]]:
        raise NotImplementedError("More code will be released once accepted")

    def domac_router(
        self,
        dec_ct32: np.ndarray,
        dec_ct22: np.ndarray,
        gamma,
        lr,
        train_steps,
        bm_steps,
        device,
        weight_wns,
        weight_tns,
        weight_L_D,
        weight_L_BM,
        weight_L_inc,
        tb_logger: torch.utils.tensorboard.writer.SummaryWriter = None,
        json_file=None,
        load_from_json=None,
        **kwargs,
    ):
        raise NotImplementedError("More code will be released once accepted")

    def random_router(
        self, dec_ct32: np.ndarray, dec_ct22: np.ndarray
    ) -> Tuple[str, List[deque]]:
        v_src = ""
        stage_num = len(dec_ct32)
        column_num = len(self.pp)
        # wires for stage 0
        input_wire_list = []
        for column_index in range(column_num):
            wire_list = [
                f"pp_{column_index}[{wire_index}]"
                for wire_index in range(int(self.pp[column_index]))
            ]
            input_wire_list.append(deque(wire_list))
        for stage_index in range(stage_num):
            v_src += self.declare_comment(f"stage {stage_index}")
            output_wire_list = [deque() for _ in range(column_num)]
            for column_index in range(column_num):
                for ct32_index in range(int(dec_ct32[stage_index][column_index])):
                    ct32_name = f"ct32_s{stage_index}_c{column_index}_{ct32_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(3)
                    ]
                    sum_name = f"sum_from_{ct32_name}"
                    carry_name = f"carry_from_{ct32_name}"
                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_fa(
                        ct32_name, input_wires, sum_name, carry_name
                    )
                for ct22_index in range(int(dec_ct22[stage_index][column_index])):
                    ct22_name = f"ct22_s{stage_index}_c{column_index}_{ct22_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(2)
                    ]
                    sum_name = f"sum_from_{ct22_name}"
                    carry_name = f"carry_from_{ct22_name}"
                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_ha(
                        ct22_name, input_wires, sum_name, carry_name
                    )
                all_wires = []
                while len(output_wire_list[column_index]) > 0:
                    wire = output_wire_list[column_index].pop()
                    all_wires.append(wire)
                while len(input_wire_list[column_index]) > 0:
                    wire = input_wire_list[column_index].pop()
                    all_wires.append(wire)
                np.random.shuffle(all_wires)
                for wire in all_wires:
                    input_wire_list[column_index].appendleft(wire)
        v_src += "\n"
        routed_wire_list = input_wire_list
        return v_src, routed_wire_list

    def declare_wire(self, wire_name):
        return f"    wire {wire_name};\n"

    def declare_fa(self, fa_name, input_list, sum_name, carry_name):
        v_src = f"    FA {fa_name} (\n"
        v_src += f"        .a ({input_list[0]}),\n"
        v_src += f"        .b({input_list[1]}),\n"
        v_src += f"        .cin({input_list[2]}),\n"
        v_src += f"        .sum({sum_name}),\n"
        v_src += f"        .cout({carry_name})\n"
        v_src += "    );\n"
        return v_src

    def declare_ha(self, ha_name, input_list, sum_name, carry_name):
        v_src = f"    HA {ha_name} (\n"
        v_src += f"        .a ({input_list[0]}),\n"
        v_src += f"        .cin({input_list[1]}),\n"
        v_src += f"        .sum({sum_name}),\n"
        v_src += f"        .cout({carry_name})\n"
        v_src += "    );\n"
        return v_src

    def declare_comment(self, comment: str, num_stars=50, align="c"):
        v_src = "/*" + "*" * num_stars + "\n"
        lines = comment.splitlines()
        for line in lines:
            if len(line) > 0:
                if align == "c":
                    v_src += " * " + line.center(num_stars - 2) + "*\n"
                elif align == "l":
                    v_src += " * " + line.ljust(num_stars - 2) + "*\n"
                else:
                    v_src += " * " + line.rjust(num_stars - 2) + "*\n"
        v_src += "*" * num_stars + "*/\n"
        return v_src

    def emit_verilog_backup(self, file_path=None, dec_ct32=None, dec_ct22=None):
        if dec_ct32 is None or dec_ct22 is None:
            dec_ct32, dec_ct22 = self.compressor_assignment()
        stage_num = len(dec_ct32)
        column_num = len(self.pp)

        comment = ""
        comment += f"Generated by CompressorTree\n"
        comment += f"ct32={self.ct32}\n"
        comment += f"ct22={self.ct22}\n"
        comment += f"dec_ct32=\n{dec_ct32}\n"
        comment += f"dec_ct22=\n{dec_ct22}\n"
        v_src = self.declare_comment(comment, align="l")

        v_src += "module CompressorTree(\n"
        input_pp_list = []
        for column_index in range(len(self.pp)):
            pp_height = self.pp[column_index]
            if pp_height > 0:
                v_src += f"    input [{pp_height-1}:0] pp_{column_index},\n"
        v_src += ",\n".join(input_pp_list)
        v_src += f"    output [{column_num-1}:0] out"
        v_src += "\n);\n"

        # wires for stage 0
        input_wire_list = []
        for column_index in range(column_num):
            wire_list = [
                f"pp_{column_index}[{wire_index}]"
                for wire_index in range(int(self.pp[column_index]))
            ]
            input_wire_list.append(deque(wire_list))
        for stage_index in range(stage_num):
            v_src += self.declare_comment(f"stage {stage_index}")
            output_wire_list = [deque() for _ in range(column_num)]
            for column_index in range(column_num):
                for ct32_index in range(int(dec_ct32[stage_index][column_index])):
                    ct32_name = f"ct32_s{stage_index}_c{column_index}_{ct32_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(3)
                    ]
                    sum_name = f"sum_from_{ct32_name}"
                    carry_name = f"carry_from_{ct32_name}"
                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_fa(
                        ct32_name, input_wires, sum_name, carry_name
                    )
                for ct22_index in range(int(dec_ct22[stage_index][column_index])):
                    ct22_name = f"ct22_s{stage_index}_c{column_index}_{ct22_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(2)
                    ]
                    sum_name = f"sum_from_{ct22_name}"
                    carry_name = f"carry_from_{ct22_name}"
                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_ha(
                        ct22_name, input_wires, sum_name, carry_name
                    )
                while len(output_wire_list[column_index]) > 0:
                    wire = output_wire_list[column_index].pop()
                    input_wire_list[column_index].appendleft(wire)
        v_src += "\n"

        v_src += f"    wire [{column_num-1}:0] a;\n"
        v_src += f"    wire [{column_num-1}:0] b;\n"
        for column_index in range(column_num):
            input_wires = input_wire_list[column_index]
            if len(input_wires) == 1:
                wire = input_wires[0]
                v_src += f"    assign a[{column_index}] = {wire};\n"
                v_src += f"    assign b[{column_index}] = 1'b0;\n"
            elif len(input_wires) == 2:
                wire_1, wire_2 = input_wires
                v_src += f"    assign a[{column_index}] = {wire_1};\n"
                v_src += f"    assign b[{column_index}] = {wire_2};\n"
            else:
                raise ValueError

        v_src += f"    assign out = a + b;\n"
        v_src += "endmodule\n"

        if file_path is not None:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w") as f:
                f.write(v_src)
        return v_src

    def rlmul_assigner(
        self, dec_ct32: np.ndarray, dec_ct22: np.ndarray
    ) -> Tuple[str, List[deque]]:
        v_src = ""
        stage_num = len(dec_ct32)
        column_num = len(self.pp)
        input_wire_list = []
        for column_index in range(column_num):
            wire_list = [
                f"pp_{column_index}[{wire_index}]"
                for wire_index in range(int(self.pp[column_index]))
            ]
            input_wire_list.append(deque(wire_list))
        for stage_index in range(stage_num):
            v_src += self.declare_comment(f"stage {stage_index}")
            output_sum_list_32 = [deque() for _ in range(column_num)]
            output_sum_list_22 = [deque() for _ in range(column_num)]
            output_carry_list_32 = [deque() for _ in range(column_num)]
            output_carry_list_22 = [deque() for _ in range(column_num)]
            for column_index in range(column_num):
                for ct32_index in range(int(dec_ct32[stage_index][column_index])):
                    ct32_name = f"ct32_s{stage_index}_c{column_index}_{ct32_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(3)
                    ]
                    sum_name = f"sum_from_{ct32_name}"
                    carry_name = f"carry_from_{ct32_name}"
                    output_sum_list_32[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_carry_list_32[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_fa(
                        ct32_name, input_wires, sum_name, carry_name
                    )
                for ct22_index in range(int(dec_ct22[stage_index][column_index])):
                    ct22_name = f"ct22_s{stage_index}_c{column_index}_{ct22_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(2)
                    ]
                    sum_name = f"sum_from_{ct22_name}"
                    carry_name = f"carry_from_{ct22_name}"
                    output_sum_list_22[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_carry_list_22[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_ha(
                        ct22_name, input_wires, sum_name, carry_name
                    )
                while len(output_carry_list_32[column_index]) > 0:
                    wire = output_carry_list_32[column_index].pop()
                    input_wire_list[column_index].appendleft(wire)
                while len(output_carry_list_22[column_index]) > 0:
                    wire = output_carry_list_22[column_index].pop()
                    input_wire_list[column_index].appendleft(wire)
                while len(output_sum_list_22[column_index]) > 0:
                    wire = output_sum_list_22[column_index].pop()
                    input_wire_list[column_index].append(wire)
                while len(output_sum_list_32[column_index]) > 0:
                    wire = output_sum_list_32[column_index].pop()
                    input_wire_list[column_index].append(wire)
        v_src += "\n"
        routed_wire_list = input_wire_list
        return v_src, routed_wire_list

    def emit_prefix_adder(self, prefix_adder: str):
        v_src = ""
        if prefix_adder is None:
            v_src += f"    assign out = a + b;\n"
        else:
            remain_pp = self.get_remain_pp()
            cell_map = adder_utils.get_init_cell_map(len(remain_pp), prefix_adder)
            v_src += f"    // {prefix_adder} adder\n"
            v_src += adder_utils.emit_fused_verilog(cell_map, remain_pp)
        return v_src

    def emit_verilog(
        self,
        file_path: str = None,
        dec_ct32: np.ndarray = None,
        dec_ct22: np.ndarray = None,
        router: str = None,
        prefix_adder: str = None,
        n_processing=4,
        Z_constant=1000,
        timeLimit=60,
        keepFiles=False,
        method="GUROBI_CMD",
        **kwargs,
    ):
        if dec_ct32 is None or dec_ct22 is None:
            dec_ct32, dec_ct22 = self.compressor_assignment()
        column_num = len(self.pp)

        comment = ""
        comment += f"Generated by CompressorTree\n"
        comment += f"ct32={self.ct32}\n"
        comment += f"ct22={self.ct22}\n"
        comment += f"dec_ct32=\n{dec_ct32}\n"
        comment += f"dec_ct22=\n{dec_ct22}\n"
        v_src = self.declare_comment(comment, align="l")

        v_src += "module CompressorTree(\n"
        input_pp_list = []
        for column_index in range(len(self.pp)):
            pp_height = self.pp[column_index]
            if pp_height > 0:
                v_src += f"    input [{pp_height-1}:0] pp_{column_index},\n"
        v_src += ",\n".join(input_pp_list)
        v_src += f"    output [{column_num-1}:0] out"
        v_src += "\n);\n"

        if router is None or router == "None" or router == "rlmul":
            router_src, routed_wire_list = self.rlmul_assigner(dec_ct32, dec_ct22)
        elif router == "random":
            router_src, routed_wire_list = self.random_router(dec_ct32, dec_ct22)
        elif router == "ufomac":
            router_src, routed_wire_list = self.ufomac_router(
                dec_ct32,
                dec_ct22,
                method,
                Z_constant,
                n_processing,
                timeLimit,
                keepFiles,
                **kwargs,
            )
        elif router == "domac":
            router_src, routed_wire_list = self.domac_router(
                dec_ct32,
                dec_ct22,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Unknown router: {router}")
        v_src += router_src
        v_src += f"    wire [{column_num-1}:0] a;\n"
        v_src += f"    wire [{column_num-1}:0] b;\n"
        for column_index in range(column_num):
            input_wires = routed_wire_list[column_index]
            if len(input_wires) == 1:
                wire = input_wires[0]
                v_src += f"    assign a[{column_index}] = {wire};\n"
                v_src += f"    assign b[{column_index}] = 1'b0;\n"
            elif len(input_wires) == 2:
                wire_1, wire_2 = input_wires
                v_src += f"    assign a[{column_index}] = {wire_1};\n"
                v_src += f"    assign b[{column_index}] = {wire_2};\n"
            else:
                raise ValueError
        v_src += self.emit_prefix_adder(prefix_adder)
        v_src += "endmodule\n"

        if file_path is not None:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w") as f:
                f.write(v_src)
        return v_src

    def emit_verilog_fused_assignment(
        self,
        file_path: str = None,
        assignment: List[List[List[Tuple]]] = None,
        prefix_adder: str = None,
    ):
        column_num = len(self.pp)

        comment = ""
        comment += f"Generated by CompressorTree\n"
        comment += f"ct32={self.ct32}\n"
        comment += f"ct22={self.ct22}\n"
        v_src = self.declare_comment(comment, align="l")

        v_src += "module CompressorTree(\n"
        input_pp_list = []
        for column_index in range(len(self.pp)):
            pp_height = self.pp[column_index]
            if pp_height > 0:
                v_src += f"    input [{pp_height-1}:0] pp_{column_index},\n"
        v_src += ",\n".join(input_pp_list)
        v_src += f"    output [{column_num-1}:0] out"
        v_src += "\n);\n"

        input_wire_list = []
        for column_index in range(column_num):
            wire_list = [
                f"pp_{column_index}[{wire_index}]"
                for wire_index in range(int(self.pp[column_index]))
            ]
            input_wire_list.append(deque(wire_list))
        for stage_index in range(len(assignment)):
            v_src += self.declare_comment(f"stage {stage_index}")
            output_wire_list = [deque() for _ in range(column_num)]
            for column_index in range(column_num):
                for compressor_id in assignment[stage_index][column_index]:
                    s, c, t, idx = compressor_id
                    assert s == stage_index and c == column_index
                    if t == 0:
                        ct_name = f"ct32_s{stage_index}_c{column_index}_{idx}"
                        input_wires = [
                            input_wire_list[column_index].pop() for _ in range(3)
                        ]
                        declare_ct: Callable = self.declare_fa
                    elif t == 1:
                        ct_name = f"ct22_s{stage_index}_c{column_index}_{idx}"
                        input_wires = [
                            input_wire_list[column_index].pop() for _ in range(2)
                        ]
                        declare_ct: Callable = self.declare_ha
                    else:
                        raise NotImplementedError(f"Unknown compressor type: {t}")
                    sum_name = f"sum_from_{ct_name}"
                    carry_name = f"carry_from_{ct_name}"
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)

                    v_src += declare_ct(ct_name, input_wires, sum_name, carry_name)

                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                while len(output_wire_list[column_index]) > 0:
                    wire = output_wire_list[column_index].pop()
                    input_wire_list[column_index].appendleft(wire)
        v_src += "\n"
        routed_wire_list = input_wire_list

        v_src += f"    wire [{column_num-1}:0] a;\n"
        v_src += f"    wire [{column_num-1}:0] b;\n"
        for column_index in range(column_num):
            input_wires = routed_wire_list[column_index]
            if len(input_wires) == 1:
                wire = input_wires[0]
                v_src += f"    assign a[{column_index}] = {wire};\n"
                v_src += f"    assign b[{column_index}] = 1'b0;\n"
            elif len(input_wires) == 2:
                wire_1, wire_2 = input_wires
                v_src += f"    assign a[{column_index}] = {wire_1};\n"
                v_src += f"    assign b[{column_index}] = {wire_2};\n"
            else:
                raise ValueError

        v_src += self.emit_prefix_adder(prefix_adder)
        v_src += "endmodule\n"

        if file_path is not None:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w") as f:
                f.write(v_src)
        return v_src

    def emit_verilog_from_dict(
        self,
        file_path: str = None,
        assignment: dict = None,
        prefix_adder: str = None,
    ):
        column_num = len(self.pp)

        comment = ""
        comment += f"Generated by CompressorTree\n"
        comment += f"ct32={self.ct32}\n"
        comment += f"ct22={self.ct22}\n"
        v_src = self.declare_comment(comment, align="l")

        v_src += "module CompressorTree(\n"
        input_pp_list = []
        for column_index in range(len(self.pp)):
            pp_height = self.pp[column_index]
            if pp_height > 0:
                v_src += f"    input [{pp_height-1}:0] pp_{column_index},\n"
        v_src += ",\n".join(input_pp_list)
        v_src += f"    output [{column_num-1}:0] out"
        v_src += "\n);\n"

        v_src += assignment["router_src"]
        routed_wire_list = assignment["routed_wire_list"]

        v_src += f"    wire [{column_num-1}:0] a;\n"
        v_src += f"    wire [{column_num-1}:0] b;\n"
        for column_index in range(column_num):
            input_wires = routed_wire_list[column_index]
            if len(input_wires) == 1:
                wire = input_wires[0]
                v_src += f"    assign a[{column_index}] = {wire};\n"
                v_src += f"    assign b[{column_index}] = 1'b0;\n"
            elif len(input_wires) == 2:
                wire_1, wire_2 = input_wires
                v_src += f"    assign a[{column_index}] = {wire_1};\n"
                v_src += f"    assign b[{column_index}] = {wire_2};\n"
            else:
                raise ValueError

        v_src += self.emit_prefix_adder(prefix_adder)
        v_src += "endmodule\n"

        if file_path is not None:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w") as f:
                f.write(v_src)
        return v_src

    def legalize(self):
        initial_pp = copy.deepcopy(self.pp)
        ct32 = copy.deepcopy(self.ct32)
        ct22 = copy.deepcopy(self.ct22)
        assert len(ct32) == len(initial_pp) and len(ct22) == len(initial_pp)
        for column_index in range(0, len(initial_pp)):
            ct32[column_index] = max(0, ct32[column_index])
            ct22[column_index] = max(0, ct22[column_index])
            if column_index == 0:
                remain_pp = (
                    initial_pp[column_index]
                    - 2 * ct32[column_index]
                    - ct22[column_index]
                )
            else:
                remain_pp = (
                    initial_pp[column_index]
                    + ct32[column_index - 1]
                    + ct22[column_index - 1]
                    - 2 * ct32[column_index]
                    - ct22[column_index]
                )
            if remain_pp < 1:
                if ct22[column_index] + remain_pp >= 1:
                    ct22[column_index] += remain_pp - 1
                else:
                    remain_pp += ct22[column_index]
                    ct22[column_index] = 0
                    if remain_pp % 2 == 0:
                        ct32[column_index] -= (2 - remain_pp) // 2
                    else:
                        ct32[column_index] -= (1 - remain_pp) // 2
            elif remain_pp > 2:
                if remain_pp - ct22[column_index] <= 2:
                    ct22[column_index] -= remain_pp - 2
                    ct32[column_index] += remain_pp - 2
                else:
                    ct32[column_index] += ct22[column_index]
                    remain_pp -= ct22[column_index]
                    ct22[column_index] = 0
                    if remain_pp % 2 == 0:
                        ct32[column_index] += (remain_pp - 2) / 2
                    else:
                        ct32[column_index] += (remain_pp - 1) / 2

        remain_pp = copy.deepcopy(initial_pp)
        remain_pp[0] = initial_pp[0] - 2 * ct32[0] - ct22[0]
        for column_index in range(1, len(initial_pp)):
            remain_pp[column_index] = (
                initial_pp[column_index]
                + ct32[column_index - 1]
                + ct22[column_index - 1]
                - 2 * ct32[column_index]
                - ct22[column_index]
            )
        remain_pp = np.asarray(remain_pp)
        assert (remain_pp >= 1).all() and (remain_pp <= 2).all(), "legalize fail"
        self.ct32 = ct32
        self.ct22 = ct22
        return ct32, ct22

    def get_remain_pp(self):
        remain_pp = np.zeros_like(self.pp)
        carry_in = 0
        for i in range(len(self.pp)):
            remain_pp[i] = self.pp[i] + carry_in - self.ct32[i] * 2 - self.ct22[i]
            carry_in = self.ct32[i] + self.ct22[i]
        return remain_pp

    def __repr__(self):
        return f"ct32={self.ct32}, ct22={self.ct22}"
