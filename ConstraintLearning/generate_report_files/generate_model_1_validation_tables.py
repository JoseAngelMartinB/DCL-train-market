import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path for imports
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

MODEL = "Model_1"
INPUT_CSV = os.path.join(project_root, f"ConstraintLearning/validation_data/final_results_model_1.csv")
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


def build_validation_table(day_df, day):
    lines = []

    day_str = str(day)
    safe_day = day_str.replace("-", "_")
    day_label = DAY_LABELS.get(day_str, day_str)
    original_passengers = fmt_int(day_df["original_passengers_mean"].mean())

    lines.append(r"\begin{table}[!h]")
    lines.append(r"    \centering")
    lines.append(rf"    \caption{{ROBIN validation results for Model~1 on {day_str} ({day_label}). Baseline demand: {original_passengers} passengers for the incumbent operator (RENFE).}}")
    lines.append(rf"    \label{{tab:model1_validation_{safe_day}}}")
    lines.append(r"    \scriptsize")
    lines.append(r"    \setlength{\tabcolsep}{3pt}")
    #lines.append(r"    \resizebox{\textwidth}{!}{%")
    lines.append(r"    \begin{tabular}{llrrrrr}")
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{Model}")
    lines.append(r"        & \(\boldsymbol{\delta}\)")
    lines.append(r"        & \textbf{New pax}")
    lines.append(r"        & \textbf{Orig. rev.}")
    lines.append(r"        & \textbf{Actual rev.}")
    lines.append(r"        & \textbf{Rev. diff.}")
    lines.append(r"        & \textbf{(\%)} \\")
    lines.append(r"        \midrule")

    for model, model_df in day_df.groupby("model", sort=False):
        model_nrows = len(model_df)
        first_model_row = True

        for _, row in model_df.iterrows():
            model_cell = f"\\multirow{{{model_nrows}}}{{*}}{{{model}}}" if first_model_row else ""

            line = (
                f"        {model_cell} "
                f"& {row['delta']} "
                f"& {fmt_mean_ci(row['new_passengers_mean'], row['new_passengers_se'], 0)} "
                f"& {fmt_int(row['original_revenue_mean'])} "
                f"& {fmt_mean_ci(row['actual_revenue_mean'], row['actual_revenue_se'], 0)} "
                f"& {fmt_mean_ci(row['revenue_difference_mean'], row['revenue_difference_se'], 0)} "
                f"& {fmt_float(row['revenue_difference_percentage'], 2)} \\\\"
            )
            lines.append(line)

            first_model_row = False

        lines.append(r"        \addlinespace")

    if lines[-1] == r"        \addlinespace":
        lines.pop()

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    #lines.append(r"    }")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required_cols = {
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
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing_cols)}")

    df = df[df["optim_model"] == MODEL].copy()

    df["model"] = df["ml_model"].map(MODEL_MAP).fillna(df["ml_model"])

    df["model_order"] = df["model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else len(MODEL_ORDER)
    )

    df["delta_order"] = df["delta"].apply(
        lambda x: DELTA_ORDER.index(x) if x in DELTA_ORDER else len(DELTA_ORDER)
    )

    df = df.sort_values(["day", "model_order", "delta_order"])

    for day, day_df in df.groupby("day", sort=True):
        day_df = day_df.copy()

        latex = build_validation_table(day_df, day)

        safe_day = str(day).replace("-", "_")
        output_file = os.path.join(OUTPUT_DIR, f"Model_1_validation_{safe_day}.tex")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"Written: {output_file}")


if __name__ == "__main__":
    main()