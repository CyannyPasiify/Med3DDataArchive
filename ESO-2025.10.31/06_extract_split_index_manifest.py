# -*- coding: utf-8 -*-
"""
Dataset Split Index Manifest Extraction Script for ESO-2025.10.31 Dataset

This script extracts subset-specific index manifests from a main manifest Excel file based on split information.

Parameters:
    -m, --manifest_file: Path to the Excel manifest file
    -spn, --split_name: Name of the split scheme (e.g., split01_TJ)
    -o, --output_index_manifest_dir: Output directory for index manifest files
    -s, --subsets: Subset names to extract (can be specified multiple times, e.g., -s train -s val -s test)

Usage Examples:
    python 06_extract_split_index_manifest.py -m /path/to/dataset_manifest.xlsx -spn split01_TJ -o /path/to/output -s train -s test
    python 06_extract_split_index_manifest.py --manifest_file /path/to/dataset_manifest.xlsx --split_name split01_TJ --output_index_manifest_dir /path/to/output --subsets train val
    python 06_extract_split_index_manifest.py -m /path/to/dataset_manifest.xlsx -spn split01_TJ -o /path/to/output -s train -s val -s test
"""

import argparse
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional, Union, Any, Set


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing manifest_file, split_name, output_index_manifest_dir, and subsets
    """
    parser = argparse.ArgumentParser(
        description='Extract subset-specific index manifests from main manifest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m /path/to/dataset_manifest.xlsx -spn split01_TJ -o /path/to/output -s train -s test
  %(prog)s --manifest_file /path/to/dataset_manifest.xlsx --split_name split01_TJ --output_index_manifest_dir /path/to/output --subsets train val
  %(prog)s -m /path/to/dataset_manifest.xlsx -spn split01_TJ -o /path/to/output -s train -s val -s test
        """
    )

    parser.add_argument(
        '-m', '--manifest_file',
        type=str,
        required=True,
        help='Path to the Excel manifest file'
    )

    parser.add_argument(
        '-spn', '--split_name',
        type=str,
        required=True,
        help='Name of the split scheme (e.g., split01_TJ)'
    )

    parser.add_argument(
        '-o', '--output_index_manifest_dir',
        type=str,
        required=True,
        help='Output directory for index manifest files'
    )

    parser.add_argument(
        '-s', '--subsets',
        type=str,
        nargs='+',
        action='append',
        help='Subset names to extract (can be specified multiple times, e.g., -s train -s val -s test)'
    )

    return parser.parse_args()


def validate_subsets(df: pd.DataFrame, split_name: str, subset_groups: List[List[str]]) -> Tuple[bool, List[str]]:
    """
    Validate that all specified subsets exist in the split column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the split information
        split_name (str): Name of the split column
        subset_groups (List[List[str]]): List of subset groups to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, invalid_subsets) where invalid_subsets is a list of invalid subset names
    """
    if split_name not in df.columns:
        print(f"Error: Split column '{split_name}' not found in manifest")
        return False, []

    valid_subsets: Set[str] = set(df[split_name].dropna().unique())
    invalid_subsets: List[str] = []

    for group in subset_groups:
        for subset in group:
            if subset not in valid_subsets:
                invalid_subsets.append(subset)

    return len(invalid_subsets) == 0, invalid_subsets


def extract_subset_manifest(df: pd.DataFrame, split_name: str, subsets: List[str]) -> pd.DataFrame:
    """
    Extract rows matching the specified subsets from the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the split information
        split_name (str): Name of the split column
        subsets (List[str]): List of subset names to extract
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only the specified subsets
    """
    mask = df[split_name].isin(subsets)
    return df[mask].copy()


def save_index_manifest(df: pd.DataFrame, output_dir: Union[str, Path], split_name: str, subsets: List[str]) -> Path:
    """
    Save the filtered DataFrame to an Excel file.
    
    Args:
        df (pd.DataFrame): Filtered DataFrame to save
        output_dir (Union[str, Path]): Output directory path
        split_name (str): Name of the split scheme
        subsets (List[str]): List of subset names
        
    Returns:
        Path: Path to the saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{split_name}_{'_'.join(sorted(subsets))}.xlsx"
    sheetname = f"{split_name}_{'_'.join(sorted(subsets))}"
    file_path = output_path / filename

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheetname)

    return file_path


def process_split_index_manifest(
    manifest_file: Union[str, Path], 
    split_name: str, 
    output_index_manifest_dir: Union[str, Path], 
    subset_groups: Optional[List[List[str]]]
) -> None:
    """
    Process the manifest file and extract index manifests for each subset group.
    
    Args:
        manifest_file (Union[str, Path]): Path to the Excel manifest file
        split_name (str): Name of the split scheme
        output_index_manifest_dir (Union[str, Path]): Output directory for index manifest files
        subset_groups (Optional[List[List[str]]]): List of subset groups to extract
    """
    manifest_path = Path(manifest_file)
    output_path = Path(output_index_manifest_dir)

    if not manifest_path.exists():
        print(f"Error: Manifest file does not exist: {manifest_file}")
        return

    print(f"Reading manifest file: {manifest_file}")
    print(f"Split name: {split_name}")
    print(f"Output directory: {output_index_manifest_dir}")

    try:
        df = pd.read_excel(
            manifest_path, 
            sheet_name=split_name, 
            dtype={'ID': str, 'site': str, 'pid': str}
        )

        print(f"Total records in manifest: {len(df)}")

        if not subset_groups:
            print("Warning: No subset groups specified. Nothing to extract.")
            return

        is_valid, invalid_subsets = validate_subsets(df, split_name, subset_groups)

        if not is_valid:
            print(f"\nError: The following subsets do not exist in the '{split_name}' column:")
            for subset in invalid_subsets:
                print(f"  - {subset}")
            print("\nValid subsets in the manifest:")
            valid_subsets = sorted(df[split_name].dropna().unique())
            for subset in valid_subsets:
                print(f"  - {subset}")
            return

        print(f"\nExtracting index manifests for {len(subset_groups)} subset group(s):\n")

        for i, subsets in enumerate(subset_groups, 1):
            print(f"[{i}/{len(subset_groups)}] Processing subset group: {', '.join(sorted(subsets))}")

            filtered_df = extract_subset_manifest(df, split_name, subsets)
            count = len(filtered_df)

            print(f"  Found {count} records")

            file_path = save_index_manifest(filtered_df, output_path, split_name, subsets)
            print(f"  Saved to: {file_path}\n")

        print("Index manifest extraction completed successfully!")

    except Exception as e:
        print(f"Error processing manifest file: {str(e)}")
        return


def main() -> None:
    """
    Main function to orchestrate the split index manifest extraction process.
    """
    args: argparse.Namespace = parse_args()

    print(f"Processing manifest file: {args.manifest_file}")
    print(f"Split name: {args.split_name}")
    print(f"Output directory: {args.output_index_manifest_dir}")

    if args.subsets:
        print(f"Subset groups to extract:")
        for i, group in enumerate(args.subsets, 1):
            print(f"  Group {i}: {', '.join(sorted(group))}")
    else:
        print("Warning: No subset groups specified")

    process_split_index_manifest(args.manifest_file, args.split_name, args.output_index_manifest_dir, args.subsets)

    print("Split index manifest extraction completed successfully!")


if __name__ == '__main__':
    main()