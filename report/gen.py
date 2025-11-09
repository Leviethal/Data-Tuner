import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Use current folder (report/) instead of looking for "report" inside it
ROOT = "."

# Create main image output folder if not exists
os.makedirs("../report_img", exist_ok=True)

def parse_file(path):
    with open(path, "r") as f:
        text = f.read()

    missing_rate = float(re.findall(r"_(0\.\d+)", os.path.basename(path))[0])

    # Parse MAE
    mae_section = text.split("=== Mean Absolute Error (MAE) per Numeric Column ===")[1]
    mae_section = mae_section.split("\n\n")[0].strip().split("\n")
    mae = {line.split(":")[0].strip(): float(line.split(":")[1]) for line in mae_section}

    # Parse Accuracy
    try:
        acc_section = text.split("=== Accuracy")[1]
        acc_section = acc_section.split("\n\n")[0].strip().split("\n")[1:]
        acc = {line.split(":")[0].strip(): float(line.split(":")[1].replace("%", "").strip()) for line in acc_section}
    except:
        acc = {}

    # Parse Mean Values (patched to preserve full column name)
    mean_section = text.split("=== Mean values after imputation ===")[1]
    mean_section = mean_section.split("\n\n")[0].strip().split("\n")

    mean_vals = {}
    for line in mean_section:
        if "dtype" in line:
            continue
        col, val = line.rsplit(maxsplit=1)  # ← key fix here
        mean_vals[col.strip()] = float(val)

    # Normalize MAE safely
    norm_mae = {}
    for col in mae:
        if col in mean_vals and mean_vals[col] != 0:
            norm_mae[col] = mae[col] / mean_vals[col]
        else:
            norm_mae[col] = None

    return missing_rate, mae, acc, norm_mae


def process_dataset(dataset_folder):
    dataset_name = os.path.basename(dataset_folder)

    for file in sorted(os.listdir(dataset_folder)):
        if not file.endswith(".txt"):
            continue
        
        path = os.path.join(dataset_folder, file)
        missing, mae, acc, norm = parse_file(path)
        missing_percent = int(missing * 100)

        # ✅ Output saved to: ../report_img/<dataset>/<missing>/
        out_dir = os.path.join("..", "report_img", dataset_name, f"{missing}")
        os.makedirs(out_dir, exist_ok=True)

        # ---- Create Combined Plot ----
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ---- Plot MAE ----
        mae_items = [(k, v) for k, v in mae.items() if v is not None]
        if mae_items:
            cols, vals = zip(*mae_items)
            axes[0].bar(cols, vals, color="steelblue")
        axes[0].set_title("MAE (Numeric Columns)")
        axes[0].set_ylabel("MAE")
        axes[0].tick_params(axis='x', rotation=45)

        # ---- Plot Accuracy ----
        acc_items = [(k, v) for k, v in acc.items() if v is not None]
        if acc_items:
            cols, vals = zip(*acc_items)
            axes[1].bar(cols, vals, color="darkgreen")
        axes[1].set_title("Categorical Accuracy (%)")
        axes[1].set_ylabel("Accuracy %")
        axes[1].tick_params(axis='x', rotation=45)

        # ---- Plot Normalized MAE ----
        norm_items = [(k, v) for k, v in norm.items() if v is not None]
        if norm_items:
            cols, vals = zip(*norm_items)
            axes[2].bar(cols, vals, color="firebrick")
        axes[2].set_title("Normalized MAE (MAE / Mean)")
        axes[2].set_ylabel("Normalized MAE")
        axes[2].tick_params(axis='x', rotation=45)


        fig.suptitle(f"{dataset_name} - {missing_percent}% Missing Data", fontsize=16)
        fig.tight_layout()

        save_path = os.path.join(out_dir, f"{dataset_name}_{missing_percent}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"✅ Saved: {save_path}")


def main():
    for dataset in os.listdir(ROOT):
        dataset_folder = os.path.join(ROOT, dataset)
        if os.path.isdir(dataset_folder):
            process_dataset(dataset_folder)


if __name__ == "__main__":
    main()
