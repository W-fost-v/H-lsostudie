from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "health_study_dataset.csv"

columns = ["age", "weight", "height", "systolic_bp", "cholesterol"]

def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def describe(df, columns):
    return df[columns].agg(["mean", "median", "min", "max"])


def plot_hist(df, column, bins=20):
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=bins, edgecolor="black")
    plt.title(f"Fördelning av {column}")
    plt.xlabel(column)
    plt.ylabel("Antal deltagare")
    plt.show()

def plot_box_by_group(df, value_col, group_col):
    groups = sorted(df[group_col].unique())
    data = [df.loc[df[group_col] == g, value_col] for g in groups]

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, tick_labels=[str(g) for g in groups])
    plt.title(f"{value_col} by {group_col}")
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.show()


def plot_share_bar(df, column):
    share = df[column].value_counts(normalize=True) * 100

    plt.figure(figsize=(6, 5))
    plt.bar(share.index.astype(str), share.values)
    plt.title(f"Andel per kategori: {column}")
    plt.xlabel(column)
    plt.ylabel("Andel av deltagare (%)")
    plt.show()


def disease_share(df):  
    return df["disease"].mean()

def simulate_disease(p, n, rng):
    return rng.binomial(1, p, size=n)

def ci_mean_normal(x):
    n = len(x)
    mean = x.mean()
    std_err = x.std(ddof=1) / (n ** 0.5)
    z_score = 1.96  
    ci_lower = mean - z_score * std_err
    ci_upper = mean + z_score * std_err
    return ci_lower, ci_upper


def mean_diff_test(a, b, n_boot, rng):
   
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    observed = np.mean(a) - np.mean(b)
 
    overall_mean = np.mean(np.concatenate([a, b]))
    a0 = a - np.mean(a) + overall_mean
    b0 = b - np.mean(b) + overall_mean

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        a_sample = rng.choice(a0, size=len(a0), replace=True)
        b_sample = rng.choice(b0, size=len(b0), replace=True)
        boot_diffs[i] = np.mean(a_sample) - np.mean(b_sample)

    p_value = np.mean(boot_diffs >= observed)
    return float(observed), float(p_value), boot_diffs

def plot_mean_diffs(boot_diffs, observed):
   
    plt.figure(figsize=(8, 5))
    plt.hist(boot_diffs, bins=40, edgecolor="black")
    plt.axvline(observed, color="red", label="Observerad skillnad")
    plt.title("Bootstrap-test: skillnad i medelblodtryck under H0")
    plt.xlabel("Skillnad rökare - icke-rökare")
    plt.ylabel("Antal bootstrap-omsamplingar")
    plt.legend()
    plt.show()