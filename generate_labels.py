import os
import pandas as pd

# Root path to your ISIC 2018 dataset on the server
ISIC_ROOT = "/home/jayastp/data/ISIC2018_Task3"

# Correct full paths to the CSV label files inside their subdirectories
csv_paths = {
    "train": os.path.join(ISIC_ROOT, "ISIC2018_Task3_Training_GroundTruth", "ISIC2018_Task3_Training_GroundTruth.csv"),
    "val":   os.path.join(ISIC_ROOT, "ISIC2018_Task3_Validation_GroundTruth", "ISIC2018_Task3_Validation_GroundTruth.csv"),
    "test":  os.path.join(ISIC_ROOT, "ISIC2018_Task3_Test_GroundTruth", "ISIC2018_Task3_Test_GroundTruth.csv"),
}

# Class label mapping: maps class name to integer index
class_labels = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
label_to_index = {label: i for i, label in enumerate(class_labels)}

def convert_csv(split_name, csv_path):
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        img_id = row["image"]
        for label in class_labels:
            if row[label] == 1:
                class_idx = label_to_index[label]
                records.append((img_id, class_idx))
                break
    df_out = pd.DataFrame(records, columns=["image_id", "label"])
    out_path = os.path.join("data", f"{split_name}_labels.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_out)} rows)")

# Create output folder and process all splits
os.makedirs("data", exist_ok=True)
for split, path in csv_paths.items():
    convert_csv(split, path)
