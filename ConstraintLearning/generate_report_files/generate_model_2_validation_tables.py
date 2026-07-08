import pandas as pd
import os
import sys

# Add project root to sys.path for imports
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


INPUT_CSV = os.path.join(project_root, "ConstraintLearning/validation_data/final_results_model_2_C.csv")
OUTPUT_DIR = os.path.join(project_root, "ConstraintLearning/latex_tables")

T_CRIT_95_N25 = 2.064

MODEL_MAP = {
    "tree": "DT",
    "rf": "RF",
    "gbm": "GBDT",
    "ffnn": "FFNN",
}

MODEL_ORDER = ["DT", "RF", "GBDT", "FFNN"]
DELTA_ORDER = [5, 10, 20]

DAY_LABELS = {
    "2025-03-12": "low-season weekday",
    "2025-03-22": "low-season weekend",
    "2025-08-13": "high-season weekday",
    "2025-08-23": "high-season weekend",
}

REQUIRED_COLS = {
    "optim_model",
    "ml_model",
    "delta",
    "day",
    "original_passengers_mean",
    "new_passengers_mean",
    "new_passengers_se",
    "original_revenue_mean",
    "actual_revenue_mean",
    "actual_revenue_se",
    "revenue_difference_mean",
    "revenue_difference_se",
    "revenue_difference_percentage",
    "original_passengers_mean_competitors",
    "new_passengers_mean_competitors",
    "new_passengers_se_competitors",
    "actual_revenue_mean_competitors",
    "actual_revenue_se_competitors",
    "revenue_difference_percentage_competitors",
}


def clean_model_name(x):
    if pd.isna(x):
        return "--"
    x = str(x)
    return MODEL_MAP.get(x, x)


def parse_model_tuple(x):
    if pd.isna(x):
        return "-- + --"

    x = str(x)

    if "-" not in x:
        clean_name = clean_model_name(x)
        return f"{clean_name} + {clean_name}"

    demand_raw, price_raw = x.split("-", 1)
    demand_raw = demand_raw.replace("demand_", "")
    price_raw = price_raw.replace("price_", "")

    demand_model = clean_model_name(demand_raw)
    price_model = clean_model_name(price_raw)

    return f"{demand_model} + {price_model}"


def fmt_float(x, digits=2):
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_int(x):
    if pd.isna(x):
        return "--"
    return f"{int(round(x)):,}"


def fmt_mean_ci(mean, se, digits=2):
    if pd.isna(mean):
        return "--"

    if pd.isna(se):
        if digits == 0:
            return fmt_int(mean)
        return fmt_float(mean, digits)

    ci = T_CRIT_95_N25 * se

    if digits == 0:
        return rf"{fmt_int(mean)} $\pm$ {fmt_int(ci)}"

    return rf"{fmt_float(mean, digits)} $\pm$ {fmt_float(ci, digits)}"


def add_ordering_columns(df):
    df = df.copy()

    df["model_order"] = df["model_label"].apply(
        lambda x: MODEL_ORDER.index(x.split("+")[0].strip())
        if x.split("+")[0].strip() in MODEL_ORDER
        else len(MODEL_ORDER)
    )

    df["delta_order"] = df["delta"].apply(
        lambda x: DELTA_ORDER.index(x) if x in DELTA_ORDER else len(DELTA_ORDER)
    )

    return df.sort_values(["day", "model_order", "delta_order"])


def load_validation_results(df, optim_model):
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing_cols)}")

    df = df[df["optim_model"] == optim_model].copy()

    if optim_model == "Model_1":
        df["model_label"] = df["ml_model"].apply(clean_model_name)
    elif optim_model == "Model_2":
        df["model_label"] = df["ml_model"].apply(parse_model_tuple)
    else:
        raise ValueError(f"Unsupported optimization model: {optim_model}")

    return add_ordering_columns(df)


def append_result_rows(lines, day_df, label_offset=""):
    for model_label, model_df in day_df.groupby("model_label", sort=False):
        model_nrows = len(model_df)
        first_model_row = True

        for _, row in model_df.iterrows():
            model_cell = (
                f"\\multirow{{{model_nrows}}}{{*}}{{{model_label}}}"
                if first_model_row
                else ""
            )

            line = (
                f"        {model_cell} "
                f"& {row['delta']} "
                f"& {fmt_mean_ci(row['new_passengers_mean'], row['new_passengers_se'], 0)} "
                f"& {fmt_int(row['original_revenue_mean'])} "
                f"& {fmt_mean_ci(row['actual_revenue_mean'], row['actual_revenue_se'], 0)} "
                f"& {fmt_mean_ci(row['revenue_difference_mean'], row['revenue_difference_se'], 0)} "
                f"& {fmt_float(row['revenue_difference_percentage'], 2)} "
                f"& {fmt_mean_ci(row['new_passengers_mean_competitors'], row['new_passengers_se_competitors'], 0)} "
                f"& {fmt_mean_ci(row['actual_revenue_mean_competitors'], row['actual_revenue_se_competitors'], 0)} "
                f"& {fmt_float(row['revenue_difference_percentage_competitors'], 2)} \\\\"
            )
            lines.append(line)

            first_model_row = False

        lines.append(r"        \addlinespace")

    if lines[-1] == r"        \addlinespace":
        lines.pop()


def build_validation_table(day_df_model_1, day_df_model_2, day):
    lines = []

    day_str = str(day)
    safe_day = day_str.replace("-", "_")
    day_label = DAY_LABELS.get(day_str, day_str)

    original_passengers = fmt_int(day_df_model_1["original_passengers_mean"].mean())
    competitor_passengers = fmt_int(day_df_model_1["original_passengers_mean_competitors"].mean())
    baseline_text = (
        f"Baseline demand: {original_passengers} passengers for the incumbent operator (RENFE) "
        f"and {competitor_passengers} passengers for the aggregated competitors."
    )

    lines.append(r"\begin{table}[!h]")
    lines.append(r"    \centering")
    lines.append(rf"    \caption{{ROBIN validation results under competitor reaction on {day_str} ({day_label}). {baseline_text}}}")
    lines.append(rf"    \label{{tab:model2_validation_{safe_day}}}")
    lines.append(r"    \scriptsize")
    lines.append(r"    \setlength{\tabcolsep}{3pt}")
    lines.append(r"    \begin{tabular}{llrrrrrrrr}")
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{Model spec.}")
    lines.append(r"        & \(\boldsymbol{\delta}\)")
    lines.append(r"        & \textbf{New pax}")
    lines.append(r"        & \textbf{Orig. rev.}")
    lines.append(r"        & \textbf{Actual rev.}")
    lines.append(r"        & \textbf{Rev. diff.}")
    lines.append(r"        & \textbf{(\%)}")
    lines.append(r"        & \textbf{Comp. pax}")
    lines.append(r"        & \textbf{Comp. rev.}")
    lines.append(r"        & \textbf{diff. (\%)} \\")
    lines.append(r"        \midrule")

    lines.append(r"        \multicolumn{10}{l}{\textbf{Model~1:}} \\[1ex]")
    append_result_rows(lines, day_df_model_1)

    lines.append(r"        \midrule")
    lines.append(r"        \multicolumn{10}{l}{\textbf{Model~2:}} \\")
    append_result_rows(lines, day_df_model_2)

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_all = pd.read_csv(INPUT_CSV)

    df_model_1 = load_validation_results(df_all, "Model_1")
    df_model_2 = load_validation_results(df_all, "Model_2")

    days = sorted(set(df_model_1["day"]).intersection(set(df_model_2["day"])))

    for day in days:
        day_df_model_1 = df_model_1[df_model_1["day"] == day].copy()
        day_df_model_2 = df_model_2[df_model_2["day"] == day].copy()

        latex = build_validation_table(day_df_model_1, day_df_model_2, day)

        safe_day = str(day).replace("-", "_")
        output_file = os.path.join(OUTPUT_DIR, f"Model_2_validation_{safe_day}.tex")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"Written: {output_file}")


if __name__ == "__main__":
    main()