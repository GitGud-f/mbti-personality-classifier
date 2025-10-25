"""
Script to run full data preprocessing and save processed data.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from mbti_personality_classifier.data.processing_pipeline import run_full_preprocessing


def main():
    parser = argparse.ArgumentParser(description="Preprocess MBTI dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw CSV file (overrides default data/raw/mbti_1.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/processed_data.pkl",
        help="Output path for processed data (Pickle format)"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    df_processed = run_full_preprocessing(filepath=args.input)

    # Save
    print(f"💾 Saving processed data to {output_path}...")
    df_processed.to_pickle(output_path)
    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    main()