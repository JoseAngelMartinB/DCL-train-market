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
INPUT_CSV = os.path.join(project_root, f"ConstraintLearning/{MODEL}/opt_results.csv")
OUTPUT_TEX = os.path.join(project_root, f"ConstraintLearning/latex_tables/{MODEL}_optimization_table.tex")

MODEL_MAP = {
    "tree": "DT",
    "rf": "RF",
    "gbm": "GBDT",
    "ffnn": "FFNN",
}

MODEL_ORDER = ["DT", "RF", "GBDT", "FFNN"]

DAY_ORDER = ["2025-03-12", "2025-03-22", "2025-08-13", "2025-08-23"]


def format_status(status_series):
    status_series = status_series.dropna().astype(str)

    if status_series.empty:
        return "--"

    unique_status = status_series.unique()

    if len(unique_status) == 1:
        return unique_status[0]

    n_optimal = (status_series == "Optimal").sum()
    n_total = len(status_series)
    return f"{n_optimal}/{n_total} Opt."


def fmt_float(x, digits=2):
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_int(x):
    if pd.isna(x):
        return "--"
    return f"{int(round(x)):,}"


def fmt_status(x):
    if pd.isna(x):
        return "--"
    x = str(x)
    return {
        "Optimal": "Opt.",
        "Time Limit": "TL",
        "TimeLimit": "TL",
        "Infeasible": "Inf.",
        "Feasible": "Feas.",
    }.get(x, x)


def build_latex_table(df):
    lines = []

    lines.append(r"\begin{table}[!h]")
    lines.append(r"    \centering")
    lines.append(r"        \caption{}")
    lines.append(r"        \label{}")
    lines.append(r"        \scriptsize")
    lines.append(r"        \setlength{\tabcolsep}{3pt}")
    #lines.append(r"        \resizebox{\textwidth}{!}{%")
    lines.append(r"        \begin{tabular}{llrrrrrrrr}")
    lines.append(r"            \toprule")
    lines.append(r"            \textbf{Model} ")
    lines.append(r"            & \(\boldsymbol{\delta}\)")
    lines.append(r"            & \textbf{Day}")
    lines.append(r"            & \textbf{Status}")
    lines.append(r"            & \textbf{Obj.}")
    lines.append(r"            & \textbf{Gap (\%)}")
    lines.append(r"            & \textbf{Time (s)}")
    lines.append(r"            & \textbf{Vars.}")
    lines.append(r"            & \textbf{Constr.}")
    lines.append(r"            & \textbf{Mem. (GB)} \\")
    lines.append(r"            \midrule")

    for model, model_df in df.groupby("model", sort=False):
        model_nrows = len(model_df)
        first_model_row = True

        for delta, delta_df in model_df.groupby("delta", sort=False):
            delta_nrows = len(delta_df)
            first_delta_row = True

            for _, row in delta_df.iterrows():
                model_cell = f"\\multirow{{{model_nrows}}}{{*}}[-1.5ex]{{{model}}}" if first_model_row else ""
                delta_cell = f"\\multirow{{{delta_nrows}}}{{*}}{{{row['delta']}}}" if first_delta_row else ""

                line = (
                    f"            {model_cell} "
                    f"& {delta_cell} "
                    f"& {row['day']} "
                    f"& {fmt_status(row['status'])} "
                    f"& {fmt_float(row['expected_revenue'], 2)} "
                    f"& {fmt_float(row['gap_pct'], 2)} "
                    f"& {fmt_float(row['time_seconds'], 2)} "
                    f"& {fmt_int(row['n_vars'])} "
                    f"& {fmt_int(row['n_constrs'])} "
                    f"& {fmt_float(row['peak_ram_usage_GB'], 2)} \\\\"
                )
                lines.append(line)

                first_model_row = False
                first_delta_row = False

            lines.append(r"            \addlinespace")

        lines.append(r"            \midrule")

    if lines[-1] == r"            \midrule":
        lines.pop()

    lines.append(r"            \bottomrule")
    lines.append(r"        \end{tabular}")
    #lines.append(r"        }")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = {
        "ml_model",
        "day",
        "delta",
        "total_revenue",
        "gap",
        "time_seconds",
        "n_vars",
        "n_constrs",
        "peak_ram_usage_MB",
        "status",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing_cols)}")

    df["model"] = df["ml_model"].map(MODEL_MAP).fillna(df["ml_model"])

    grouped = (
        df.groupby(["model", "delta", "day"], as_index=False)
        .agg(
            expected_revenue=("total_revenue", "mean"),
            gap=("gap", "mean"),
            time_seconds=("time_seconds", "mean"),
            n_vars=("n_vars", "mean"),
            n_constrs=("n_constrs", "mean"),
            peak_ram_usage_MB=("peak_ram_usage_MB", "mean"),
            peak_ram_usage_GB=("peak_ram_usage_MB", lambda x: x.mean() / 1024),
            status=("status", format_status),
        )
    )

    # Convert solver gap to percentage.
    grouped["gap_pct"] = 100 * grouped["gap"]

    grouped["model_order"] = grouped["model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else len(MODEL_ORDER)
    )

    grouped["day_order"] = grouped["day"].apply(
        lambda x: DAY_ORDER.index(str(x)) if str(x) in DAY_ORDER else len(DAY_ORDER)
    )

    grouped = grouped.sort_values(["model_order", "delta", "day_order"]).drop(
        columns=["model_order", "day_order"]
    )

    latex = build_latex_table(grouped)

    # Check if the output directory exists, and create it if it doesn't
    if not os.path.exists(os.path.dirname(OUTPUT_TEX)):
        os.makedirs(os.path.dirname(OUTPUT_TEX), exist_ok=True)

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"\nLaTeX table written to: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()