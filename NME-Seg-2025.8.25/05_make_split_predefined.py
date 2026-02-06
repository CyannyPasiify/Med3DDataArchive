# -*- coding: utf-8 -*-
"""
Dataset Split Generation Script for NME-Seg-2025.8.25 Dataset

This script creates a new worksheet in the manifest Excel file with split information (train/val/test)
based on the subset column value and filters samples by specified sites and collections.

Parameters:
    -m, --manifest_file: Path to the Excel manifest file
    -s, --sheet_name: Name of the main manifest worksheet (default: Manifest)
    -spn, --split_name: Name of the split scheme (default: split01_Tongji)
    -s, --site: List of site codes to include (default: ['Tongji'])
    -c, --collection: List of collections to include (default: ['train-val'])

Usage Examples:
    python 05_make_split_predefined.py -m /path/to/dataset_manifest.xlsx
    python 05_make_split_predefined.py --manifest_file /path/to/dataset_manifest.xlsx -s Manifest -spn split01_Tongji
    python 05_make_split_predefined.py -m /path/to/dataset_manifest.xlsx --site Tongji OtherSite --collection train-val test
"""

import argparse
from pathlib import Path
import pandas as pd
from typing import List, Optional, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing manifest_file, sheet_name, split_name, site, and collection
    """
    parser = argparse.ArgumentParser(
        description='Create split worksheet in manifest Excel file with site and collection filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m /path/to/dataset_manifest.xlsx
  %(prog)s --manifest_file /path/to/dataset_manifest.xlsx -s Manifest -spn split01_Tongji
  %(prog)s -m /path/to/dataset_manifest.xlsx --site Tongji OtherSite --collection train-val test
        """
    )

    parser.add_argument(
        '-m', '--manifest_file',
        type=str,
        required=True,
        help='Path to the Excel manifest file'
    )

    parser.add_argument(
        '-sn', '--sheet_name',
        type=str,
        default='Manifest',
        help='Name of the main manifest worksheet (default: Manifest)'
    )

    parser.add_argument(
        '-spn', '--split_name',
        type=str,
        default='split01_Tongji',
        help='Name of the split scheme (default: split01_Tongji)'
    )

    parser.add_argument(
        '-s', '--site',
        type=str,
        nargs='+',
        default=['Tongji'],
        help='List of site codes to include (default: Tongji)'
    )

    parser.add_argument(
        '-c', '--collection',
        type=str,
        nargs='+',
        default=['train-val'],
        help='List of collections to include (default: train-val)'
    )

    return parser.parse_args()


def determine_split(subset_value: Optional[str]) -> str:
    """
    Determine split (train/val/test) based on the subset column value.
    
    Args:
        subset_value (Optional[str]): Subset value from the manifest
        
    Returns:
        str: 'train', 'val', or 'test' based on the subset value
    """
    if pd.isna(subset_value) or subset_value == '':
        return ''

    subset_lower = str(subset_value).lower()

    if subset_lower == 'train':
        return 'train'
    elif subset_lower == 'val':
        return 'val'
    elif subset_lower == 'test':
        return 'test'
    else:
        return ''


def filter_by_site_and_collection(df: pd.DataFrame, sites: List[str], collections: List[str]) -> pd.DataFrame:
    """
    Filter samples by inclusion in specified sites and collections.
    
    Args:
        df (pd.DataFrame): DataFrame containing manifest data
        sites (List[str]): List of site codes to include
        collections (List[str]): List of collections to include
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only samples from specified sites and collections
    """
    filtered_df = df.copy()
    
    # Filter by site
    if sites:
        filtered_df = filtered_df[filtered_df['site'].isin(sites)]
    
    # Filter by collection
    if collections:
        filtered_df = filtered_df[filtered_df['collection'].isin(collections)]
    
    return filtered_df


def create_split_worksheet(manifest_file: Union[str, Path], sheet_name: str, split_name: str, sites: List[str], collections: List[str]) -> None:
    """
    Create a new worksheet with split information in the manifest Excel file,
    filtered by specified sites and collections.
    
    Args:
        manifest_file (Union[str, Path]): Path to the Excel manifest file
        sheet_name (str): Name of the main manifest worksheet to copy
        split_name (str): Name of the new split worksheet
        sites (List[str]): List of site codes to include
        collections (List[str]): List of collections to include
    """
    manifest_path = Path(manifest_file)

    if not manifest_path.exists():
        print(f"Error: Manifest file does not exist: {manifest_file}")
        return

    print(f"Reading manifest file: {manifest_file}")
    print(f"Main worksheet: {sheet_name}")
    print(f"Creating split worksheet: {split_name}")
    print(f"Including sites: {', '.join(sites)}")
    print(f"Including collections: {', '.join(collections)}")

    try:
        df = pd.read_excel(
            manifest_path, 
            sheet_name=sheet_name, 
            dtype={'ID': str, 'site': str, 'pid': str}
            )

        # Check required columns
        if 'subset' not in df.columns:
            print(f"Error: 'subset' column not found in worksheet '{sheet_name}'")
            return
            
        if 'site' not in df.columns:
            print(f"Error: 'site' column not found in worksheet '{sheet_name}'")
            return
            
        if 'collection' not in df.columns:
            print(f"Error: 'collection' column not found in worksheet '{sheet_name}'")
            return

        # Filter by site and collection
        filtered_df = filter_by_site_and_collection(df, sites, collections)
        
        if filtered_df.empty:
            print(f"Error: No samples found matching the filtering criteria")
            print(f"  Sites: {', '.join(sites)}")
            print(f"  Collections: {', '.join(collections)}")
            return
            
        print(f"Filtered to {len(filtered_df)} samples from {len(df)} total samples")

        # Create copy with split information
        df_copy = filtered_df.copy()
        df_copy.insert(0, split_name, df_copy['subset'].apply(determine_split))

        train_count = (df_copy[split_name] == 'train').sum()
        val_count = (df_copy[split_name] == 'val').sum()
        test_count = (df_copy[split_name] == 'test').sum()
        unknown_count = (df_copy[split_name] == '').sum()

        print(f"\nSplit statistics:")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"  Test samples: {test_count}")
        print(f"  Unknown samples: {unknown_count}")

        if unknown_count > 0:
            unknown_samples = df_copy[df_copy[split_name] == '']
            print(f"\nWarning: {unknown_count} samples could not be classified:")
            for idx, row in unknown_samples.head(10).iterrows():
                print(f"  - ID: {row.get('ID', 'N/A')}, subset: {row.get('subset', 'N/A')}")
            if unknown_count > 10:
                print(f"  ... and {unknown_count - 10} more")

        with pd.ExcelWriter(manifest_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_copy.to_excel(writer, index=False, sheet_name=split_name)

        print(f"\nSplit worksheet '{split_name}' created successfully!")
        print(f"Total samples in split: {len(df_copy)}")

    except Exception as e:
        print(f"Error processing manifest file: {str(e)}")
        return


def main() -> None:
    """
    Main function to orchestrate the split worksheet creation process.
    """
    args = parse_args()

    print(f"Processing manifest file: {args.manifest_file}")
    print(f"Main worksheet name: {args.sheet_name}")
    print(f"Split name: {args.split_name}")
    print(f"Sites: {args.site}")
    print(f"Collections: {args.collection}")

    create_split_worksheet(args.manifest_file, args.sheet_name, args.split_name, args.site, args.collection)

    print("Split worksheet generation completed successfully!")


if __name__ == '__main__':
    main()