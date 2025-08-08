#!/usr/bin/env python3
"""
Data extraction module for LUMI vs Laptop benchmarking.
Extracts timing and accuracy data from Wandb CSV exports.
"""
import pandas as pd
from pathlib import Path


class WandbDataExtractor:
    def __init__(self, csv_dir):
        self.csv_dir = Path(csv_dir)
        self.data = []

    def extract_from_csv_files(self):
        if not self.csv_dir.exists():
            print(f"CSV directory not found: {self.csv_dir}")
            return []
        csv_files = list(self.csv_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {self.csv_dir}")
            return []
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    run_data = self._process_csv_row(row)
                    if run_data:
                        all_data.append(run_data)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        self.data = all_data
        return all_data

    def _process_csv_row(self, row):
        name = str(row.get("Name", "Unknown"))
        if name == "Unknown" or pd.isna(name):
            return None

        runtime_seconds = None
        runtime_str = str(row.get("Runtime", ""))
        if runtime_str and runtime_str != "nan":
            try:
                runtime_seconds = float(runtime_str)
            except:
                pass

        # Get nodes from CSV data
        nodes = None
        nodes_val = row.get("nodes", "")
        if pd.notna(nodes_val) and str(nodes_val) != "" and str(nodes_val) != "nan":
            try:
                nodes = int(float(nodes_val))
            except:
                pass

        # Hard-coded GPU mappings for specific run names (cleaner approach)
        gpu_mappings = {
            "1 GPU 1 Node Run": 1,
            "2 GPU 1 Node Run": 2,
            "4 GPU 1 Node Run": 4,
            "8 GPU 1 Node Run": 8,
            "16 GPU 2 Node Run": 16,
            "24 GPU 3 Node Run": 24,
            "32 GPU 4 Node Run": 32,
            "Four Node Run": 32,
            "Eight Node Run": 64,
            "Eight Node Run 2.0": 64,
            "Eight Node Run 2.1": 64,
            "Single Node Run": 8,
            "Two Node Run": 16,
            "Sixteen Node Run": 128,
            "Laptop-Multicore-2.0pct-8cores": 0,
            "Laptop-Benchmark-1.0pct-subset": 0,
        }

        # Get total_gpus from CSV data first, fallback to mappings
        total_gpus = None
        gpu_val = row.get("total_gpus", "")
        if pd.notna(gpu_val) and str(gpu_val) != "" and str(gpu_val) != "nan":
            try:
                total_gpus = int(float(gpu_val))
            except:
                pass

        if total_gpus is None and name in gpu_mappings:
            total_gpus = gpu_mappings[name]

        # Get device type from CSV data
        device_type = str(row.get("device_type", "")).upper()
        if pd.isna(device_type) or device_type == "NAN" or device_type == "":
            hardware = str(row.get("hardware", ""))
            device = str(row.get("device", ""))
            if "cpu" in hardware.lower() or "cpu" in device.lower() or "Laptop" in name:
                device_type = "CPU"
            else:
                device_type = "GPU"

        extrapolated_time = None
        extrap_val = row.get("extrapolated_total_time_hr", "")
        if pd.notna(extrap_val) and str(extrap_val) != "":
            try:
                extrapolated_time = float(extrap_val)
            except:
                pass

        # Get accuracy from CSV data
        accuracy = None
        for acc_col in ["acc", "val_accuracy", "train_accuracy"]:
            acc_val = row.get(acc_col, "")
            if pd.notna(acc_val) and str(acc_val) != "" and str(acc_val) != "nan":
                try:
                    accuracy = float(acc_val)
                    break
                except:
                    continue

        run_data = {
            "run_name": name,
            "runtime_seconds": runtime_seconds,
            "extrapolated_total_time_hr": extrapolated_time,
            "nodes": nodes,
            "total_gpus": total_gpus,
            "device_type": device_type,
            "accuracy": accuracy,
            "state": str(row.get("State", "Unknown")),
            "created": str(row.get("Created", "")),
        }
        return run_data

    def create_dataframe(self):
        df = pd.DataFrame(self.data)
        if df.empty:
            print("No data extracted from CSV files")
            return df

        df["training_time_hours"] = df.apply(self._calculate_training_hours, axis=1)
        df["is_laptop"] = df["device_type"].str.contains("CPU", case=False, na=False)
        df["is_extrapolated"] = df["extrapolated_total_time_hr"].notna()

        # Filter to valid runs with training time data
        df = df[df["training_time_hours"].notna()]

        # Separate LUMI and laptop runs
        lumi_runs = df[~df["is_laptop"]].copy()
        laptop_runs = df[df["is_laptop"]].copy()

        # Select best runs for each category
        selected_lumi = self._select_best_lumi_runs(lumi_runs)
        selected_laptop = self._deduplicate_runs(laptop_runs)

        final_df = pd.concat([selected_lumi, selected_laptop], ignore_index=True)
        return final_df

    def save_to_benchmark_csv(self, df, output_path="charts/benchmark_data.csv"):
        """Save the processed dataframe to benchmark_data.csv format"""
        if df.empty:
            print("No data to save")
            return

        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Ensure we have all required columns for benchmark_data.csv
        required_columns = [
            "run_name",
            "runtime_seconds",
            "extrapolated_total_time_hr",
            "nodes",
            "total_gpus",
            "device_type",
            "accuracy",
            "state",
            "created",
            "training_time_hours",
            "is_laptop",
            "is_extrapolated",
        ]

        # Add any missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        # Reorder columns to match benchmark_data.csv format
        df_ordered = df[required_columns]

        # Load existing data if file exists
        if output_file.exists():
            existing_df = pd.read_csv(output_file)
            print(f"Found existing benchmark data with {len(existing_df)} rows")

            # Remove duplicates based on run_name
            existing_names = set(existing_df["run_name"].tolist())
            new_data = df_ordered[~df_ordered["run_name"].isin(existing_names)]

            if len(new_data) > 0:
                # Combine and save
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                combined_df.to_csv(output_file, index=False)
                print(f"Added {len(new_data)} new rows. Total: {len(combined_df)} rows")
            else:
                print("No new data to add (all runs already exist)")
        else:
            # Save new file
            df_ordered.to_csv(output_file, index=False)
            print(f"Created new benchmark_data.csv with {len(df_ordered)} rows")

    def update_benchmark_from_csvs(self):
        """Extract data from CSV files and update benchmark_data.csv"""
        print("Extracting data from CSV files...")
        self.extract_from_csv_files()

        if not self.data:
            print("No data extracted from CSV files")
            return

        print(f"Processing {len(self.data)} runs...")
        df = self.create_dataframe()

        if df.empty:
            print("No valid data after processing")
            return

        print(f"Saving {len(df)} processed runs to benchmark_data.csv...")
        self.save_to_benchmark_csv(df)
        return df

    def _select_best_lumi_runs(self, lumi_df):
        if lumi_df.empty:
            return lumi_df

        # Only include finished runs - no fallback to other states
        finished_runs = lumi_df[lumi_df["state"] == "finished"].copy()
        if finished_runs.empty:
            print("No finished LUMI runs found")
            return pd.DataFrame()

        selected_runs = []

        # Group by total_gpus - include ALL unique GPU counts
        gpu_counts = finished_runs["total_gpus"].dropna().unique()
        gpu_counts = sorted([int(x) for x in gpu_counts if pd.notna(x)])

        for gpu_count in gpu_counts:
            gpu_runs = finished_runs[finished_runs["total_gpus"] == gpu_count]
            if not gpu_runs.empty:
                # Prefer runs with accuracy data
                valid_runs = gpu_runs[gpu_runs["accuracy"].notna()]
                if valid_runs.empty:
                    valid_runs = gpu_runs

                # Select best run based on shortest training time
                best_run = valid_runs.loc[valid_runs["training_time_hours"].idxmin()]
                selected_runs.append(best_run.to_dict())

        # Return all runs - no cap, no preferred filtering
        # Just sort by GPU count for consistent ordering
        final_df = pd.DataFrame(selected_runs)
        if not final_df.empty:
            final_df = final_df.sort_values("total_gpus")

        return final_df

    def _deduplicate_runs(self, runs_df):
        if runs_df.empty:
            return runs_df

        # For duplicate run names, keep the one with shortest training time
        df_dedup = []
        for run_name in runs_df["run_name"].unique():
            subset = runs_df[runs_df["run_name"] == run_name]
            if len(subset) > 1:
                best_run = subset.loc[subset["training_time_hours"].idxmin()]
                df_dedup.append(best_run)
            else:
                df_dedup.append(subset.iloc[0])
        return pd.DataFrame(df_dedup)

    def _calculate_training_hours(self, row):
        if pd.notna(row["extrapolated_total_time_hr"]):
            return row["extrapolated_total_time_hr"]
        elif pd.notna(row["runtime_seconds"]):
            return row["runtime_seconds"] / 3600
        else:
            return None


if __name__ == "__main__":
    # Initialize extractor with CSV directory
    extractor = WandbDataExtractor("CSVs")

    # Update benchmark_data.csv with new CSV files
    df = extractor.update_benchmark_from_csvs()

    if df is not None:
        print("\nDataFrame summary:")
        print(f"Total runs: {len(df)}")
        print(f"Device types: {df['device_type'].value_counts().to_dict()}")
        print(f"Node counts: {df['nodes'].value_counts().to_dict()}")
        if "total_gpus" in df.columns:
            print(f"GPU counts: {df['total_gpus'].value_counts().to_dict()}")
        print("\nFirst few rows:")
        display_cols = [
            "run_name",
            "nodes",
            "device_type",
            "training_time_hours",
            "accuracy",
        ]
        if "total_gpus" in df.columns:
            display_cols.insert(2, "total_gpus")
        print(df[display_cols].head())
