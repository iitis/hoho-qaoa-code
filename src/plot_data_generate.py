""" Script for evaluating true energy data 
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np

# -------------- Parameters ----------------
n = 10
layer = 10

no_experiments = range(1, 100 + 1)

weight = "random"
init_alpha_list = np.arange(0.0, 1.0 + 0.0001, 0.05)
energy_eval_for = ["step_parameter", "init_alpha"]
initialization = ["rand-rand", "nearzero-rand", "zero-rand"]
initialization = [initialization[2]]
graph_type = ["barabasi-albert", "erdos-renyi"]
possibilities = [["false", "false", "optim"]]
solver = "gd"
graph_type = graph_type[0]

# -------------- Plots selection ----------------
plot_data_list = ["init_alpha", "init_alpha_minimum_energy", "alpha_step"]
plot_data = plot_data_list[2]


# -------------Make plots-------------------------------------

if plot_data == "init_alpha":

    for init_type_no, init_type in enumerate(initialization):
        rr, nzr, zr = [], [], []  # HACK
        for pos in possibilities:
            period, bound, optwith = pos[0], pos[1], pos[2]

            for no, init_alpha in enumerate(init_alpha_list):
                init_alpha = round(init_alpha, 8)

                x_axis_alpha = alpha_range = np.arange(
                    init_alpha, 1.0 + 0.0001, 0.01
                )
                y_axis_en_energy = np.zeros(len(x_axis_alpha))

                for i in no_experiments:

                    true_en = np.load(
                        f"compare_data/data/{graph_type}/true_energy_with_init_qubit{n}-exp{i}-initalpha0-step0.01-weight{weight}.npy",
                        allow_pickle=True,
                    )[()]
                    mc_qaoa_en_full_list = np.load(
                        f"compare_data/data/{graph_type}/info_for_mc_qaoa_with_qubit{n}-layer{layer}-initalpha-{init_alpha}-inittyp-{init_type}-alpha_step0.01-exp{i}-software{optwith}-solver{solver}-weight{weight}-periodic{period}-bounds{bound}.npy"
                    )["energies"]
                    minima_list = np.asarray(
                        true_en["energy"]["minima"][5 * no :]
                    )
                    maxima_list = np.asarray(
                        true_en["energy"]["maxima"][5 * no :]
                    )

                    assert len(minima_list) == len(
                        mc_qaoa_en_full_list
                    ) and len(maxima_list) == len(minima_list)

                    numerator = mc_qaoa_en_full_list - minima_list
                    denominator = maxima_list - minima_list
                    mc_qaoa_norm_en_list = numerator / denominator
                    y_axis_en_energy += mc_qaoa_norm_en_list

                mc_qaoa_norm_en_full_list = y_axis_en_energy / len(
                    no_experiments
                )

                print(mc_qaoa_norm_en_full_list)
                file_name_norm_en = f"plot_data/{graph_type}/qubit{n}_layer{layer}_norm_energy_init_alpha-{init_alpha}_{init_type}_tot-exp-{len(no_experiments)}.npy"
                file_exists = os.path.exists(f"{file_name_norm_en}")

                if file_exists:
                    print(file_name_norm_en)
                    print("This file exists so I am skipping")
                else:
                    print("saving alert")
                    print(file_name_norm_en)
                    np.save(file_name_norm_en, mc_qaoa_norm_en_full_list)


elif plot_data == "init_alpha_minimum_energy":

    for init_type_no, init_type in enumerate(initialization):
        for pos in possibilities:
            period, bound, optwith = pos[0], pos[1], pos[2]
            y_axis = []  # HACK
            for no, init_alpha in enumerate(init_alpha_list):
                init_alpha = round(init_alpha, 8)
                for_the_sake_of_std = []

                x_axis_alpha = np.arange(init_alpha, 1 + 0.001, 0.001)
                y_axis_en_energy = 0

                for i in no_experiments:
                    print(i)
                    true_en = np.load(
                        f"compare_data/data/{graph_type}/true_energy_with_init_qubit{n}-exp{i}-initalpha0-step0.01-weight{weight}.npy",
                        allow_pickle=True,
                    )[()]
                    mc_qaoa_en_full_list = np.load(
                        f"compare_data/data/{graph_type}/info_for_mc_qaoa_with_qubit{n}-layer{layer}-initalpha-{init_alpha}-inittyp-{init_type}-alpha_step0.01-exp{i}-software{optwith}-solver{solver}-weight{weight}-periodic{period}-bounds{bound}.npy"
                    )["energies"][-1]
                    minima_list = true_en["energy"]["minima"][-1]
                    maxima_list = true_en["energy"]["maxima"][-1]

                    numerator = mc_qaoa_en_full_list - minima_list
                    denominator = maxima_list - minima_list
                    mc_qaoa_norm_en_list = numerator / denominator
                    for_the_sake_of_std.append(mc_qaoa_norm_en_list)
                    y_axis_en_energy += mc_qaoa_norm_en_list

                mc_qaoa_optimal_en_full_list = y_axis_en_energy / len(
                    no_experiments
                )
                print(mc_qaoa_optimal_en_full_list)
                y_axis.append(mc_qaoa_optimal_en_full_list)
                std_optimal = np.std(for_the_sake_of_std)
                file_name_norm_en = f"plot_data/{graph_type}/qubit{n}_norm_optimal_energy_init_alpha-{init_alpha}_{init_type}_tot-exp-{len(no_experiments)}.npy"
                file_name_std = f"plot_data/{graph_type}/qubit{n}_optimal_std_init_alpha-{init_alpha}_{init_type}_tot-exp-{len(no_experiments)}.npy"
                file_exists_norm = os.path.exists(f"{file_name_norm_en}")
                file_exists_std = os.path.exists(f"{file_name_std}")

                if file_exists_norm and file_exists_std:
                    print("The following file is already saved")
                    print(file_name_norm_en)
                    print("and")
                    print(file_name_std)
                    print()
                else:
                    print("saving alert")
                    print(file_name_std)
                    print("and")
                    print(file_name_std)
                    np.save(file_name_std, std_optimal)
                    np.save(file_name_norm_en, mc_qaoa_optimal_en_full_list)
                    print()
    plt.plot(y_axis)
    plt.show()

elif plot_data == "alpha_step":

    for init_type_no, init_type in enumerate(initialization):
        for pos in possibilities:
            period, bound, optwith = pos[0], pos[1], pos[2]
            # y_axis, x_axis, y_axis_min = [], [],[] # HACK
            for no, alpha_step in enumerate(
                [
                    1.0,
                    1 / 2,
                    1 / 4,
                    1 / 6,
                    1 / 10,
                    1 / 20,
                    1 / 40,
                    1 / 60,
                    1 / 80,
                    1 / 100,
                    1 / 200,
                    1 / 400,
                    1 / 600,
                    1 / 1000,
                    1 / 2000,
                    1 / 6000,
                    1 / 10000,
                ]
            ):
                print(alpha_step)
                for_the_sake_of_std = []
                y_axis_en_energy = 0
                a, b = 0, 0
                for i in no_experiments:
                    # print(i)
                    mc_qaoa_en_full_list = np.load(
                        f"compare_data/data/{graph_type}/info_for_mc_qaoa_with_qubit{n}-layer{layer}-initalpha-0.0-inittyp-{init_type}-alpha_step{alpha_step}-exp{i}-software{optwith}-solver{solver}-weight{weight}-periodic{period}-bounds{bound}.npy"
                    )["energies"][-1]
                    true_en = np.load(
                        f"compare_data/data/{graph_type}/true_energy_with_step_qubit{n}-exp{i}-initalpha0.0-step{alpha_step}-weight{weight}.npy",
                        allow_pickle=True,
                    )[()]

                    minima_list = np.asarray(true_en["energy"]["minima"][-1])
                    maxima_list = np.asarray(true_en["energy"]["maxima"][-1])

                    numerator = mc_qaoa_en_full_list - minima_list

                    denominator = maxima_list - minima_list
                    mc_qaoa_norm_en_list = numerator / denominator
                    for_the_sake_of_std.append(mc_qaoa_norm_en_list)
                    y_axis_en_energy += mc_qaoa_norm_en_list
                    assert (
                        numerator > 0
                    ), f"{numerator, mc_qaoa_en_full_list, minima_list}"
                    assert denominator > 0

                mc_qaoa_optimal_en_full_list = y_axis_en_energy / len(
                    no_experiments
                )
                print(mc_qaoa_optimal_en_full_list)

                std_optimal = np.std(for_the_sake_of_std)
                file_name_norm_en = f"plot_data/{graph_type}/qubit{n}_layer{layer}_norm_optimal_energy_alpha_step-{alpha_step}_{init_type}_tot-exp-{len(no_experiments)}.npy"
                file_name_std = f"plot_data/{graph_type}/qubit{n}_layer{layer}_optimal_std_alpha_step-{alpha_step}_{init_type}_tot-exp-{len(no_experiments)}.npy"
                file_exists_norm = os.path.exists(f"{file_name_norm_en}")
                file_exists_std = os.path.exists(f"{file_name_std}")

                if file_exists_norm and file_exists_std:
                    print("The following file is already saved")
                    print(file_name_norm_en)
                    print("and")
                    print(file_name_std)
                    print()
                else:
                    print("saving alert")
                    print(file_name_std)
                    print("and")
                    print(file_name_std)
                    np.save(file_name_std, std_optimal)
                    np.save(file_name_norm_en, mc_qaoa_optimal_en_full_list)
                    print()

                    print(1 / alpha_step, alpha_step)

            plt.show()
