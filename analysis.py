import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "mathtext.fontset": "custom",
    "axes.labelsize": 14,  # Tamanho da fonte dos rótulos dos eixos
    "legend.fontsize": 14,  # Tamanho da fonte das legendas
    "xtick.labelsize": 12,  # Tamanho da fonte dos rótulos do eixo x
    "ytick.labelsize": 12,  # Tamanho da fonte dos rótulos do eixo y
})



def read_mat(filename):
    with h5py.File(filename, "r") as f:
        data = np.asarray(f["Sub_score"])
    return data


def plot_time(metric, depth, target_window):
    subjects = [i for i in range(1, 31)]
    time_windows = [f"[0, {i*500}]" for i in range(1, 5)]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    mean_values = []
    std_values = []

    for time_window in time_windows:
        metric_values = []
        for subject in subjects:
            dataset = pd.read_csv(
                f"results/subject_{subject}_depth_{depth}_targets_{target_window}.csv"
            )
            dataset = dataset[dataset["time_window"] == time_window]
            metric_values.append(dataset[metric].mean())
        metric_values = np.array(metric_values)
        # exit()
        values_mean = metric_values.mean(axis=0)   
        mean_values.append(values_mean)
        std_values.append(metric_values.std(axis=0))

    mean_values = np.array(mean_values)
    std_values = np.array(std_values)

    time_windows = [float(time_window.strip("[]").split(",")[1])/1000 for time_window in time_windows]
    plt.plot(
        time_windows,
        mean_values,
        color="black",
        label="Média",
    )
    plt.fill_between(
        time_windows,
        mean_values - std_values,
        mean_values + std_values,
        color="black",
        alpha=0.1,
        label="Desvio padrão",
    )

    ax1.set_xlim(0.4, 2)
    # ax1.tick_params(axis="x", labelrotation=90)
    if metric == "accuracy":
        ax1.set_ylim(0, 1)
        
    # ax1.set_xlim(0, 60 - target_window)
    ax1.set_xlabel("Tempo (s)")
    if metric == "itr":
        ax1.set_ylabel("ITR (bits/min)")
    else:
        ax1.set_ylabel("Acurácia")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()



def plot_metric(metric, depth, target_window):
    subjects = [i for i in range(1, 31)]
    hilo = {
        "low": 0,
        "high": 1,
    }
    shades_of_blue = ["#B3C7D6", "#8AA9C1", "#5E8CB3", "#2E6E9E"]
    time_windows = [f"[0, {i*500}]" for i in range(1, 5)]

    fig, ax1 = plt.subplots(figsize=(11, 4.1))
    comfort_data = read_mat("data/Sub_score.mat")
    comfort_data = comfort_data[0, :, hilo[depth], :]
    comfort_data = comfort_data.mean(axis=1)
    comfort_data_mean = np.convolve(
        comfort_data, np.ones((target_window,)) / target_window, mode="valid"
    )
    for time_window in time_windows:
        values = []
        for subject in subjects:
            # load dataset
            dataset = pd.read_csv(
                f"results/subject_{subject}_depth_{depth}_targets_{target_window}_itr.csv"
            )
            dataset = dataset[dataset["time_window"] == time_window]
            values.append(dataset[metric])
        values = np.array(values)
        values_mean = values.mean(axis=0)
        color_idx = time_windows.index(time_window)
        if depth == "low":
            ax1.plot(
                dataset["target"],
                values_mean,
                color=shades_of_blue[color_idx],
                label=f"Tempo: {time_window.strip('[]').split(',')[1]}ms",
            )
        else:
            ax1.plot(
                dataset["target"],
                values_mean,
                color=shades_of_blue[color_idx],
                label=f"Tempo: {time_window.strip("[]").split(",")[1]}ms",
            )
            # plot the std deviation
            # std = values.std(axis=0)
            # ax1.fill_between(
            #     dataset["target"],
            #     values_mean - std,
            #     values_mean + std,
            #     color=shades_of_blue[color_idx],
            #     alpha=0.1,
            # )

            # print correlation between comfort and metric
            corr = np.corrcoef(comfort_data_mean, values_mean)[0, 1]
            print(f"Correlation between comfort and {metric}: {corr}")
        ax1.tick_params(axis="x", labelrotation=90)
        if metric == "accuracy":
            ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(comfort_data_mean, color="red", label="Conforto", zorder=0)
    ax2.set_ylim(0, 5)
    if metric == "accuracy":
        ax1.axhline(
            y=1 / target_window,
            color="black",
            linestyle="--",
            alpha=0.5,
            label="Aleatório",
        )
    ax1.set_xlim(0, 60 - target_window)
    ax1.set_xlabel("Alvos")
    if metric == "itr":
        ax1.set_ylabel("ITR (bits/min)")
    else:
        ax1.set_ylabel("Acurácia")
    ax2.set_ylabel("Conforto")
    # fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
    if(target_window == 2):
        plt.legend(handles1 + handles2, labels1 + labels2, loc="lower right")


    plt.tight_layout()
    plt.savefig(
        f"plots/{metric}_{depth}_targets_{target_window}_pls.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()


if __name__ == "__main__":
    subjects = [i for i in range(1, 31)]
    target_window = 16
    depth = "high"
    metric = "accuracy"
    time_windows = [f"[0, {i*500}]" for i in range(1, 5)]
    shades_of_blue = ["#B3C7D6", "#8AA9C1", "#5E8CB3", "#2E6E9E"]

    hilo = {
        "low": 0,
        "high": 1,
    }

    # CONFORTO E METRICA

    for depth in ["low", "high"]:
        for target_window in [2, 4, 8, 16]:
            plot_metric("itr", depth, target_window)
            plot_metric("accuracy", depth, target_window)
    exit()


    # TEMPO E METRICA

    for depth in ["low", "high"]:
        for target_window in [2, 4, 8, 16]:
            # plot_time("itr", depth, target_window)
            plot_time("accuracy", depth, target_window)
    exit()


    # PARETO TUDO

    target_windows = [2, 4, 8, 16]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    all_values = []
    for depth in ["low", "high"]:
        comfort_data = read_mat("data/Sub_score.mat")
        comfort_data = comfort_data[0, :, hilo[depth], :]
        comfort_data = comfort_data.mean(axis=1)
        for target_window in target_windows:
            targets = [
                (i, i + target_window - 1)
                for i in range(1, 60 + 1 - target_window + 1)
            ]
            comfort_data_mean = np.convolve(
                comfort_data,
                np.ones((target_window,)) / target_window,
                mode="valid",
            )
            for time_window in time_windows:
                values = []
                for subject in subjects:
                    dataset = pd.read_csv(
                        f"results/subject_{subject}_depth_{depth}_targets_{target_window}.csv"
                    )
                    dataset = dataset[dataset["time_window"] == time_window]
                    values.append(dataset[metric])
                values = np.array(values)

                values = values.mean(axis=0)

                for idx, value in enumerate(values):
                    all_values.append(
                        (
                            (depth, targets[idx], time_window.strip("[]").split(",")[1]),
                            (5 - comfort_data_mean[idx], value),
                        )
                    )

    for i in range(len(all_values)):
        plt.scatter(
            all_values[i][1][0],
            all_values[i][1][1],
            color="lightgrey",
            marker="o" if all_values[i][0][0] == "low" else "^",
            zorder=1,
        )

    non_dom = []
    for i in range(len(all_values)):
        dominated = False
        for j in range(len(all_values)):
            if i == j:
                continue
            if (
                all_values[i][1][0] >= all_values[j][1][0]
                and all_values[i][1][1] <= all_values[j][1][1]
            ):
                dominated = True
                break
        if not dominated:
            non_dom.append(all_values[i])

    label_high = False
    label_low = False

    for point in non_dom:
        if point[0][0] == "high":
            if not label_high:
                plt.scatter(
                    point[1][0],
                    point[1][1],
                    color="black",
                    marker="^",
                    label="Amplitude alta",
                    zorder=3,
                )
                label_high = True
            else:
                plt.scatter(
                    point[1][0],
                    point[1][1],
                    color="black",
                    marker="^",
                    zorder=3,
                )
        else:
            if not label_low:
                plt.scatter(
                    point[1][0],
                    point[1][1],
                    color="black",
                    label="Amplitude baixa",
                    zorder=3,
                )
                label_low = True
            else:
                plt.scatter(
                    point[1][0],
                    point[1][1],
                    color="black",
                    zorder=3,
                )

    # order non-dominated points by comfort and print
    non_dom = sorted(non_dom, key=lambda x: x[1][0])
    for i in range(len(non_dom)):
        print(
            f"Amplitude: {non_dom[i][0][0]}, Alvos: {non_dom[i][0][1]}, Tempo:"
            f" {non_dom[i][0][2]}"
        )
        print(f"Desconforto: {non_dom[i][1][0]}, {metric}: {non_dom[i][1][1]}")
    
    if metric == "accuracy":
        # plt.ylim((1/target_window) - 0.1, 1)
        # plt.ylim(0, None)
        plt.axhline(
            y= 1 - (1 / target_window),
            color="black",
            linestyle="--",
            alpha=0.5,
            label="Aleatório",
        )

    # Save a LaTeX table with the non-dominated points, ordered by discomfort
    with open(f"pareto_{metric}_targets_{target_window}.tex", "w") as f:
        f.write("\\begin{tabular}{c c c c c}\n")
        f.write("\\hline\n")
        f.write("Amplitude & Alvos & Tempo (ms) & Desconforto & Erro \\\\\n")
        f.write("\\hline\n")
        # Sort non-dominated points by discomfort
        non_dom_sorted = sorted(non_dom, key=lambda x: x[1][0])
        for point in non_dom_sorted:
            f.write(
                f"{point[0][0]} & {point[0][1]} & {point[0][2]} & {point[1][0]:.2f} & {point[1][1]:.2f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    # Save a LaTeX table with all points, with non-dominated points in bold, ordered by discomfort
    with open(f"pareto_{metric}_targets_{target_window}_all.tex", "w") as f:
        f.write("\\begin{tabular}{c c c c c}\n")
        f.write("\\hline\n")
        f.write("Amplitude & Alvos & Tempo (ms) & Desconforto & Erro \\\\\n")
        f.write("\\hline\n")
        # Sort all points by discomfort
        all_values_sorted = sorted(all_values, key=lambda x: x[1][0])
        for point in all_values_sorted:
            if point in non_dom:
                f.write(
                    f"\\textbf{{{point[0][0]}}} & \\textbf{{{point[0][1]}}} & \\textbf{{{point[0][2]}}} & \\textbf{{{point[1][0]:.2f}}} & \\textbf{{{point[1][1]:.2f}}} \\\\\n"
                )
            else:
                f.write(
                    f"{point[0][0]} & {point[0][1]} & {point[0][2]} & {point[1][0]:.2f} & {point[1][1]:.2f} \\\\\n"
                )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    plt.legend(loc="upper left")
    plt.grid(color="grey", linestyle="--", linewidth=0.5, zorder=2)
    plt.xlabel("Desconforto")
    plt.ylabel("Erro (1 - acurácia)")
    plt.savefig(
        f"plots/pareto_{metric}_targets_all.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
