import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import json


def cell_map_legalize(cell_map):
    input_bit = len(cell_map)
    for x in range(input_bit):
        cell_map[x, x] = 1
        cell_map[x, 0] = 1
    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                cell_map[last_y - 1, y] = 1
                last_y = y
    return cell_map


def get_default_init(input_bit):
    cell_map = np.zeros((input_bit, input_bit)).astype(int)
    for i in range(input_bit):
        cell_map[i, i] = 1
        cell_map[i, 0] = 1
    return np.array(cell_map)


def get_brent_kung_init(input_bit):
    cell_map = np.zeros((input_bit, input_bit)).astype(int)
    for i in range(input_bit):
        cell_map[i, i] = 1
        cell_map[i, 0] = 1
    t = 2
    while t < input_bit:
        for i in range(t - 1, input_bit, t):
            cell_map[i, i - t + 1] = 1
        t *= 2

    return np.array(cell_map)


def get_sklansky_init(input_bit):
    cell_map = np.zeros((input_bit, input_bit)).astype(int)
    for i in range(input_bit):
        cell_map[i, i] = 1
        t = i
        now = i
        x = 1
        level = 1
        while t > 0:
            if t % 2 == 1:
                last_now = now
                now -= x
                cell_map[i, now] = 1
                level += 1
            t = t // 2
            x *= 2
    cell_map = cell_map_legalize(cell_map)
    return np.array(cell_map)


def get_kogge_stone_init(input_bit):
    cell_map = np.zeros((input_bit, input_bit)).astype(int)

    for i in range(input_bit):
        cell_map[i, i] = 1
        cell_map[i, 0] = 1
        j = 1
        while j < i:
            j *= 2
            cell_map[i, i - (j - 1)] = 1

    cell_map = cell_map_legalize(cell_map)
    return np.array(cell_map)


def get_han_carlson_init(input_bit):

    cell_map = np.zeros((input_bit, input_bit))
    for i in range(input_bit):
        cell_map[i, i] = 1
        cell_map[i, 0] = 1

    t = 1
    while t < input_bit:
        for i in range(t, input_bit, t * 2):
            cell_map[i, i - t] = 1
        t *= 2

    t = 2
    while t < input_bit:
        for i in range(t - 1, input_bit, t * 2):
            cell_map[i, i - t + 1] = 1
        t *= 2

    cell_map = cell_map_legalize(cell_map)
    return np.array(cell_map)


def get_init_cell_map(input_bit: int, init_type: str):
    if init_type == "default":
        return get_default_init(input_bit)
    elif init_type == "brent_kung":
        return get_brent_kung_init(input_bit)
    elif init_type == "sklansky":
        return get_sklansky_init(input_bit)
    elif init_type == "kogge_stone":
        return get_kogge_stone_init(input_bit)
    elif init_type == "han_carlson":
        return get_han_carlson_init(input_bit)
    else:
        raise NotImplementedError


BLACK_CELL = """module BLACK(gik, pik, gkj, pkj, gij, pij);
    input gik, pik, gkj, pkj;
    output gij, pij;
    assign pij = pik & pkj;
    assign gij = gik | (pik & gkj);
endmodule
"""

GREY_CELL = """module GREY(gik, pik, gkj, gij);
    input gik, pik, gkj;
    output gij;
    assign gij = gik | (pik & gkj);
endmodule
"""


def adder_output_verilog_top(cell_map: np.ndarray) -> str:
    input_bit = len(cell_map)
    content = ""

    content += f"module PrefixAdder(a,b,s,cout);\n"
    content += f"\tinput [{input_bit - 1}:0] a,b;\n"
    content += f"\toutput [{input_bit - 1}:0] s;\n"
    content += "\toutput cout;\n"
    wires = set()
    for i in range(input_bit):
        wires.add(f"c{i}")

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:
                    wires.add(f"g{x}_{last_y}")
                    wires.add(f"p{x}_{last_y}")
                    wires.add(f"g{last_y - 1}_{y}")
                else:
                    wires.add(f"g{x}_{last_y}")
                    wires.add(f"p{x}_{last_y}")
                    wires.add(f"g{last_y - 1}_{y}")
                    wires.add(f"p{last_y - 1}_{y}")
                    wires.add(f"g{x}_{y}")
                    wires.add(f"p{x}_{y}")
                last_y = y

    for i in range(input_bit):
        wires.add(f"p{i}_{i}")
        wires.add(f"g{i}_{i}")
        wires.add(f"g{i}_{0}")
        wires.add(f"c{x}")
    assert 0 not in wires
    assert "0" not in wires
    content += "\twire "

    for i, wire in enumerate(wires):
        if i < len(wires) - 1:
            content += f"{wire},"
        else:
            content += f"{wire};\n"
    content += "\n"

    for i in range(input_bit):
        content += f"\tassign p{i}_{i} = a[{i}] ^ b[{i}];\n"
        content += f"\tassign g{i}_{i} = a[{i}] & b[{i}];\n"

    for i in range(1, input_bit):
        content += f"\tassign g{i}_0 = c{i};\n"

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:
                    content += f"\tGREY cell_{x}_{y}_grey(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, c{x});\n"
                else:
                    content += f"\tBLACK cell_{x}_{y}_black(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, p{last_y - 1}_{y}, g{x}_{y}, p{x}_{y});\n"
                last_y = y

    content += "\tassign s[0] = a[0] ^ b[0];\n"
    content += "\tassign c0 = g0_0;\n"
    content += f"\tassign cout = c{input_bit - 1};\n"
    for i in range(1, input_bit):
        content += f"\tassign s[{i}] = p{i}_{i} ^ c{i - 1};\n"
    content += "endmodule"
    content += "\n\n"

    return content


def adder_output_verilog_all(cell_map: np.ndarray, remain_pp: np.ndarray = None):
    if remain_pp is None:
        return BLACK_CELL + GREY_CELL + adder_output_verilog_top(cell_map)
    else:
        return (
            BLACK_CELL + GREY_CELL + adder_output_verilog_from_ct(cell_map, remain_pp)
        )


def adder_output_verilog_from_ct(cell_map: np.ndarray, final_pp: np.ndarray) -> str:
    input_bit = len(cell_map)
    content = ""

    content += f"module PrefixAdder("
    for column_index in range(len(final_pp)):
        content += f"out{column_index}_C, "
    content += f"s,cout,clock);\n"
    for column_index in range(len(final_pp)):
        content += f"\t input[{final_pp[column_index] - 1}:0] out{column_index}_C;\n"
    content += "\toutput cout;\n"
    content += "\tinput clock;\n"
    content += f"\toutput[{len(final_pp) - 1}:0] s;\n\n"
    content += f"\twire[{len(final_pp) - 1}:0] a;\n"
    content += f"\twire[{len(final_pp) - 1}:0] b;\n"
    for column_index in range(len(final_pp)):
        content += f"\t assign a[{len(final_pp)- 1 - column_index}] = out{column_index}_C[0];\n"
        if final_pp[column_index] == 1:
            content += f"\t assign b[{len(final_pp)- 1 - column_index}] = 1'b0;\n"
        else:
            content += f"\t assign b[{len(final_pp)- 1 - column_index}] = out{column_index}_C[1];\n"

    wires = set()
    for i in range(input_bit):
        wires.add(f"c{i}")

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:
                    wires.add(f"g{x}_{last_y}")
                    wires.add(f"p{x}_{last_y}")
                    wires.add(f"g{last_y - 1}_{y}")
                else:
                    wires.add(f"g{x}_{last_y}")
                    wires.add(f"p{x}_{last_y}")
                    wires.add(f"g{last_y - 1}_{y}")
                    wires.add(f"p{last_y - 1}_{y}")
                    wires.add(f"g{x}_{y}")
                    wires.add(f"p{x}_{y}")
                last_y = y

    for i in range(input_bit):
        wires.add(f"p{i}_{i}")
        wires.add(f"g{i}_{i}")
        wires.add(f"g{i}_{0}")
        wires.add(f"c{x}")
    assert 0 not in wires
    assert "0" not in wires
    content += "\twire "

    for i, wire in enumerate(wires):
        if i < len(wires) - 1:
            content += f"{wire},"
        else:
            content += f"{wire};\n"
    content += "\n"

    for i in range(input_bit):
        content += f"\tassign p{i}_{i} = a[{i}] ^ b[{i}];\n"
        content += f"\tassign g{i}_{i} = a[{i}] & b[{i}];\n"

    for i in range(1, input_bit):
        content += f"\tassign g{i}_0 = c{i};\n"

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:
                    content += f"\tGREY cell_{x}_{y}_grey(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, c{x});\n"
                else:
                    content += f"\tBLACK cell_{x}_{y}_black(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, p{last_y - 1}_{y}, g{x}_{y}, p{x}_{y});\n"
                last_y = y

    content += "\tassign s[0] = a[0] ^ b[0];\n"
    content += "\tassign c0 = g0_0;\n"
    content += f"\tassign cout = c{input_bit - 1};\n"
    for i in range(1, input_bit):
        content += f"\tassign s[{i}] = p{i}_{i} ^ c{i - 1};\n"
    content += "endmodule"
    content += "\n\n"

    return content


def get_cell_type_map(cell_map: np.ndarray, final_pp: np.ndarray) -> list:
    input_bit = len(cell_map)

    cell_type_map = np.full_like(cell_map, "", dtype=str).tolist()
    cell_out_map = np.full_like(cell_map, 0, dtype=int)

    for i in range(len(final_pp)):
        cell_out_map[i, i] = final_pp[i] - 1

    for x in range(input_bit):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if cell_out_map[x, last_y] == 0 and cell_out_map[last_y - 1, y] == 0:
                    cell_out_map[x, y] = 0
                else:
                    cell_out_map[x, y] = 1
                cell_type_map[x][
                    y
                ] = f"{cell_out_map[x, last_y]}{cell_out_map[last_y - 1, y]}"

                last_y = y

    return cell_type_map


def emit_fused_verilog(cell_map: np.ndarray, final_pp: np.ndarray) -> str:
    input_bit = len(cell_map)
    cell_type_map = get_cell_type_map(cell_map, final_pp)
    content = ""
    wires = set()
    for i in range(input_bit):
        wires.add(f"c{i}")

    for i in range(input_bit):
        wires.add(f"p{i}_{i}")
        wires.add(f"g{i}_{i}")
        wires.add(f"g{i}_{0}")
    assert 0 not in wires
    assert "0" not in wires

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                gij = f"g{x}_{y}"
                pij = f"p{x}_{y}"

                gik = f"g{x}_{last_y}"
                pik = f"p{x}_{last_y}"

                gkj = f"g{last_y - 1}_{y}"
                pkj = f"p{last_y - 1}_{y}"

                if y == 0:
                    wires.update({gij, gik, pik, gkj})
                else:
                    wires.update({pij, pik, pkj, gij, gik, gkj})
                last_y = y

    content += "\twire "
    for i, wire in enumerate(wires):
        if i < len(wires) - 1:
            content += f"{wire},"
        else:
            content += f"{wire};\n"
    content += "\n"

    for i in range(input_bit):
        content += f"\tassign p{i}_{i} = a[{i}] ^ b[{i}];\n"
        content += f"\tassign g{i}_{i} = a[{i}] & b[{i}];\n"

    for i in range(1, input_bit):
        content += f"\tassign c{i} = g{i}_0;\n"

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                gij = f"g{x}_{y}"
                pij = f"p{x}_{y}"

                gik = f"g{x}_{last_y}"
                pik = f"p{x}_{last_y}"

                gkj = f"g{last_y - 1}_{y}"
                pkj = f"p{last_y - 1}_{y}"

                if y == 0:
                    content += f"\n\t// GREY CELL {cell_type_map[x][y]} ({x}, {y})\n"
                    if cell_type_map[x][y] == "11":
                        content += f"\tassign {gij} = {gik} | ({pik} & {gkj});\n"
                    elif cell_type_map[x][y] == "10":
                        content += f"\tassign {gij} = {gik};\n"
                    elif cell_type_map[x][y] == "01":
                        content += f"\tassign {gij} = {pik} & {gkj};\n"
                    elif cell_type_map[x][y] == "00":
                        content += f"\tassign {gij} = 0;\n"
                else:
                    content += f"\n\t// BLACK CELL {cell_type_map[x][y]} ({x}, {y})\n"
                    content += f"\tassign {pij} = {pik} & {pkj};\n"
                    if cell_type_map[x][y] == "11":
                        content += f"\tassign {gij} = {gik} | ({pik} & {gkj});\n"
                    elif cell_type_map[x][y] == "10":
                        content += f"\tassign {gij} = {gik};\n"
                    elif cell_type_map[x][y] == "01":
                        content += f"\tassign {gij} = {pik} & {gkj};\n"
                    elif cell_type_map[x][y] == "00":
                        content += f"\tassign {gij} = 0;\n"
                last_y = y

    content += "\tassign out[0] = a[0] ^ b[0];\n"
    content += "\tassign c0 = g0_0;\n"
    for i in range(1, input_bit):
        content += f"\tassign out[{i}] = p{i}_{i} ^ c{i - 1};\n"
    return content


def emit_prefix_cells_verilog():
    return ""


def get_mask_map(cell_map: np.ndarray) -> np.ndarray:
    bit_width = len(cell_map)
    mask_map = np.full((2, bit_width, bit_width), False)
    for i in range(bit_width):
        for j in range(1, i):
            if cell_map[i, j] == 1:
                mask_map[0, i, j] = 0
                mask_map[1, i, j] = 1
            else:
                mask_map[0, i, j] = 1
                mask_map[1, i, j] = 0

    return mask_map


def get_level_map(cell_map: np.ndarray) -> np.ndarray:
    level_map = np.full_like(cell_map, -1)
    bit_width = len(cell_map)

    split_map = {}

    for x in range(bit_width - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                split_map[(x, y)] = last_y
                last_y = y

    def __get_level_value(x, y):
        if x == y:
            level_map[x, y] = 0
            return 0
        else:
            last_y = split_map[(x, y)]
            if level_map[x, y] >= 0:
                return level_map[x, y]
            else:
                left_level = __get_level_value(x, last_y)
                right_level = __get_level_value(last_y - 1, y)

                level = max(left_level, right_level) + 1
                level_map[x, y] = level

                return level

    for i in range(bit_width):
        level = __get_level_value(i, 0)
        level_map[i, 0] = level

    return level_map


def get_fanout_map(cell_map: np.ndarray) -> dict:
    bit_width = len(cell_map)
    fanout_map = {}

    for x in range(bit_width - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if (x, last_y) in fanout_map.keys():
                    fanout_map[(x, last_y)].append((x, y))
                else:
                    fanout_map[(x, last_y)] = [(x, y)]

                if (last_y - 1, y) in fanout_map.keys():
                    fanout_map[(last_y - 1, y)].append((x, y))
                else:
                    fanout_map[(last_y - 1, y)] = [(x, y)]

                last_y = y
    return fanout_map


def remove_tree_cell(cell_map: np.ndarray, target_x_list, target_y_list) -> np.ndarray:
    fanout_map = get_fanout_map(cell_map)
    new_cell_map = copy.deepcopy(cell_map)

    def __remove_cell(x, y):
        if x == y or y == 0:
            return
        else:
            new_cell_map[x, y] = 0
            if type(x) == torch.Tensor:
                x = int(x.cpu().flatten()[0])
                y = int(y.cpu().flatten()[0])
            for fanout_x, fanout_y in fanout_map[(x, y)]:
                __remove_cell(fanout_x, fanout_y)

    for x, y in zip(target_x_list, target_y_list):
        __remove_cell(x, y)
    new_cell_map = cell_map_legalize(new_cell_map)
    return new_cell_map


def draw_cell_map(cell_map: np.ndarray, power_mask: np.ndarray = None):
    plt.figure(figsize=[16, 10])

    bit_width = len(cell_map)
    level_map = get_level_map(cell_map)
    points = []
    points_color = []
    points_text = []
    lines = []
    max_level = np.max(level_map)

    for i in range(bit_width):
        last_j = i
        for j in range(i, -1, -1):
            if cell_map[i, j] == 1:
                points.append([bit_width - i, max_level - level_map[i, j]])
                points_text.append(f"({i}:{j})")
                if j == 0:
                    points_color.append("orange")
                else:
                    points_color.append("black")

                if j != i:
                    p_1 = [bit_width - i, max_level - level_map[i, j]]
                    p_2 = [bit_width - i, max_level - level_map[i, last_j]]
                    p_3 = [
                        bit_width - (last_j - 1),
                        max_level - level_map[last_j - 1, j],
                    ]
                    lines.append((p_1, p_2))
                    lines.append((p_1, p_3))

                last_j = j

    for line in lines:
        x, y = np.transpose(line)
        plt.plot(x, y, c="grey", alpha=0.5)

    x, y = np.transpose(points)
    if power_mask is None:
        plt.scatter(x, y, c=points_color)
    else:
        mask_color = []
        index = 0
        for i in range(bit_width):
            last_j = i
            for j in range(i, -1, -1):
                if cell_map[i, j] == 1:
                    mask_color.append(power_mask[i, j])
                    points_text[index] += f"\n{power_mask[i, j]:.2}"
                    last_j = j
                    index += 1
        plt.scatter(x, y, c=mask_color, s=100)

        i, j = np.unravel_index(np.argmax(power_mask), power_mask.shape)
        x, y = bit_width - i, max_level - level_map[i, j]
        plt.scatter(x, y, s=100, facecolors="none", edgecolors="red")

        plt.colorbar()
    for index, point in enumerate(points):
        x, y = np.transpose(point)
        plt.text(x, y - 0.5, points_text[index])

    plt.tight_layout()
    plt.show()
