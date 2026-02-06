# -*- coding: utf-8 -*-
"""
Data Pair Confirmation and Reorganization Script for AMOS22 Dataset

This script confirms and reorganizes image-mask pairs from the AMOS22 dataset, ensuring metadata consistency
and organizing files into a standardized directory structure grouped by manufacturer model.

Parameters:
    -r, --root_dir: Root directory of the dataset containing imagesTr, imagesTs, imagesVa, labelsTr, labelsTs, labelsVa subdirectories
    -o, --output_dir: Output directory where reorganized data will be saved
    -am, --archive_manifest: Path to Excel manifest file containing dataset metadata

Usage Examples:
    python 02_amos_ct_confirm_pairs.py -r /path/to/AMOS22 -o /path/to/AMOS-CT/grouped -am /path/to/archive_manifest.xlsx
    python 02_amos_ct_confirm_pairs.py --root_dir /path/to/AMOS22 --output_dir /path/to/AMOS-CT/grouped --archive_manifest /path/to/archive_manifest.xlsx
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, output_dir, and archive_manifest
    """
    parser = argparse.ArgumentParser(
        description='Confirm and reorganize image-mask pairs for AMOS22 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/AMOS22 -o /path/to/AMOS-CT/grouped -am /path/to/archive_manifest.xlsx
  %(prog)s --root_dir /path/to/AMOS22 --output_dir /path/to/AMOS-CT/grouped --archive_manifest /path/to/archive_manifest.xlsx
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing imagesTr, imagesTs, imagesVa, labelsTr, labelsTs, labelsVa subdirectories'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output directory where reorganized data will be saved'
    )

    parser.add_argument(
        '-am', '--archive_manifest',
        type=str,
        required=True,
        help='Path to Excel manifest file containing dataset metadata'
    )

    return parser.parse_args()


def load_archive_manifest(manifest_path):
    """
    Load archive manifest Excel file and create file information mapping.
    
    Args:
        manifest_path (str or Path): Path to Excel manifest file
        
    Returns:
        tuple: (file_info_mapping, metadata_mapping)
            - file_info_mapping: Dictionary mapping (subset, seq) to file information
            - metadata_mapping: Dictionary mapping (subset, seq) to sample metadata
    """
    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        raise FileNotFoundError(f"Archive manifest file not found: {manifest_path}")

    df = pd.read_excel(manifest_file, engine='openpyxl', dtype={'ID': str, 'seq': str, 'birth_date': str, 'acquisition_date': str})

    required_columns = ['file_path', 'subset', 'seq', 'type', 'manufacturer_model_name']
    metadata_columns = ['subset', 'seq', 'birth_date', 'sex', 'age', 'manufacturer_model_name', 'manufacturer', 'acquisition_date', 'site']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in archive manifest: {manifest_path}")

    valid_manufacturers = {'Aquilion ONE', 'Brilliance16', 'SOMATOM Force', 'Optima CT660', 'Optima CT540'}

    file_info_mapping = {}
    metadata_mapping = {}

    for _, row in df.iterrows():
        file_path = str(row['file_path'])
        subset = str(row['subset'])
        seq = str(row['seq'])
        file_type = str(row['type']).lower() if pd.notna(row['type']) else ''
        scanner = str(row['manufacturer_model_name']) if pd.notna(row['manufacturer_model_name']) else ''

        if scanner in valid_manufacturers:
            key = (subset, seq)
            if key not in file_info_mapping:
                file_info_mapping[key] = {'image_path': None, 'mask_path': None, 'scanner': scanner}
            
            if file_type == 'v3d':
                file_info_mapping[key]['image_path'] = file_path
            elif file_type == 'm3d':
                file_info_mapping[key]['mask_path'] = file_path

            if key not in metadata_mapping:
                metadata = {}
                for col in metadata_columns:
                    if col in df.columns:
                        value = row[col]
                        metadata[col] = str(value) if pd.notna(value) else 'UND'
                    else:
                        metadata[col] = 'UND'
                metadata_mapping[key] = metadata

    return file_info_mapping, metadata_mapping


def find_image_mask_pairs(root_dir, file_info_mapping):
    """
    Find matching image and mask file pairs from archive manifest.
    
    Args:
        root_dir (Path): Root directory path
        file_info_mapping (dict): Dictionary mapping (subset, seq) to file information
        
    Returns:
        dict: Dictionary mapping (subset, scanner) to list of tuples (seq_id, image_path, mask_path, scanner)
    """
    root_path = Path(root_dir)
    
    pairs_by_group = {}
    
    for (subset, seq), file_info in file_info_mapping.items():
        image_path = file_info['image_path']
        mask_path = file_info['mask_path']
        scanner = file_info['scanner']
        
        if image_path is None and mask_path is None:
            continue
        
        full_image_path = root_path / image_path if image_path else None
        full_mask_path = root_path / mask_path if mask_path else None
        
        if full_image_path and not full_image_path.exists():
            full_image_path = None
        if full_mask_path and not full_mask_path.exists():
            full_mask_path = None
        
        if full_image_path is None and full_mask_path is None:
            continue
        
        group_key = (subset, scanner)
        if group_key not in pairs_by_group:
            pairs_by_group[group_key] = []
        pairs_by_group[group_key].append((seq, full_image_path, full_mask_path, scanner))
    
    for group_key in pairs_by_group:
        pairs_by_group[group_key] = sorted(pairs_by_group[group_key], key=lambda x: x[0])
    
    return pairs_by_group


def check_metadata_consistency(image_meta, mask_meta):
    """
    Check if image and mask metadata (spatial_shape) are consistent.
    
    Args:
        image_meta (dict): Image metadata dictionary
        mask_meta (dict): Mask metadata dictionary
        
    Returns:
        bool: True if metadata is consistent, False otherwise
    """
    image_shape = image_meta['spatial_shape']
    mask_shape = mask_meta['spatial_shape']

    if not (image_shape == mask_shape).all():
        return False

    image_affine = image_meta['affine']
    mask_affine = mask_meta['affine']

    if not (image_affine == mask_affine).all():
        return False

    return True


def copy_metadata_to_mask(image_meta, mask_data):
    """
    Copy image metadata to mask to ensure consistency.
    
    Args:
        image_meta (dict): Image metadata dictionary
        mask_data: Mask image data array
        
    Returns:
        tuple: (mask_data_with_meta, updated_meta)
    """
    updated_meta = image_meta.copy()
    mask_data.meta = updated_meta
    return mask_data, updated_meta


def save_metadata_yaml(output_dir, seq_id, metadata):
    """
    Save sample metadata to YAML file.
    
    Args:
        output_dir (Path): Output directory for the YAML file
        seq_id (str): Sample sequence ID
        metadata (dict): Metadata dictionary
    """
    yaml_file = output_dir / f'amos_{seq_id}_info.yaml'
    
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)


def process_pairs(root_dir, output_dir, file_info_mapping, metadata_mapping):
    """
    Process all image-mask pairs and reorganize them into output directory.
    
    Args:
        root_dir (str or Path): Root directory containing imagesTr, imagesTs, imagesVa, labelsTr, labelsTs, labelsVa subdirectories
        output_dir (str or Path): Output directory for reorganized data
        file_info_mapping (dict): Dictionary mapping (subset, seq) to file information
        metadata_mapping (dict): Dictionary mapping (subset, seq) to sample metadata
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    pairs_by_group = find_image_mask_pairs(root_path, file_info_mapping)

    total_issues = 0
    total_pairs = 0
    total_images_only = 0
    total_masks_only = 0

    loader = LoadImage(image_only=False, dtype=None)

    for (subset, manufacturer), pairs in tqdm(pairs_by_group.items(), desc='Processing groups'):
        output_subdir = output_path / subset / manufacturer
        output_subdir.mkdir(parents=True, exist_ok=True)

        for seq_id, image_path, mask_path, _ in tqdm(pairs, desc=f'  {subset}/{manufacturer}', leave=False):
            sex = metadata_mapping[(subset, seq_id)]['sex']
            pair_dir = output_subdir / f'amos_{seq_id}_{sex}'
            pair_dir.mkdir(parents=True, exist_ok=True)

            if image_path and mask_path:
                try:
                    image_data, image_meta = loader(str(image_path))
                    mask_data, mask_meta = loader(str(mask_path))

                    if not check_metadata_consistency(image_meta, mask_meta):
                        print(f"Warning: Metadata mismatch for amos_{seq_id}")
                        print(f"  Image shape: {image_meta['spatial_shape']}, Mask shape: {mask_meta['spatial_shape']}")
                        print(f"  Copying image metadata to mask...")
                        mask_data, mask_meta = copy_metadata_to_mask(image_meta, mask_data)
                        total_issues += 1

                    volume_filestem = pair_dir / f'amos_{seq_id}_volume'
                    mask_filestem = pair_dir / f'amos_{seq_id}_mask'

                    saver_image = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.float32)
                    saver_mask = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.uint8)

                    saver_image(image_data, meta_data=image_meta, filename=volume_filestem)
                    saver_mask(mask_data, meta_data=mask_meta, filename=mask_filestem)

                    save_metadata_yaml(pair_dir, seq_id, metadata_mapping[(subset, seq_id)])

                    total_pairs += 1

                except Exception as e:
                    print(f"Error processing amos_{seq_id}: {str(e)}")
                    total_issues += 1
                    continue

            elif image_path and not mask_path:
                try:
                    image_data, image_meta = loader(str(image_path))

                    volume_filestem = pair_dir / f'amos_{seq_id}_volume'

                    saver_image = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.float32)
                    saver_image(image_data, meta_data=image_meta, filename=volume_filestem)

                    save_metadata_yaml(pair_dir, seq_id, metadata_mapping[(subset, seq_id)])

                    total_images_only += 1

                except Exception as e:
                    print(f"Error processing image amos_{seq_id}: {str(e)}")
                    total_issues += 1
                    continue

            elif mask_path and not image_path:
                try:
                    mask_data, mask_meta = loader(str(mask_path))

                    mask_filestem = pair_dir / f'amos_{seq_id}_mask'

                    saver_mask = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.uint8)
                    saver_mask(mask_data, meta_data=mask_meta, filename=mask_filestem)

                    save_metadata_yaml(pair_dir, seq_id, metadata_mapping[(subset, seq_id)])

                    total_masks_only += 1

                except Exception as e:
                    print(f"Error processing mask amos_{seq_id}: {str(e)}")
                    total_issues += 1
                    continue

    print(f"\nProcessing completed!")
    print(f"Total pairs processed: {total_pairs}")
    print(f"Total images only: {total_images_only}")
    print(f"Total masks only: {total_masks_only}")
    print(f"Total issues encountered: {total_issues}")


def main():
    """
    Main function to orchestrate pair confirmation and reorganization process.
    """
    args = parse_args()

    print(f"Processing dataset from: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Archive manifest: {args.archive_manifest}")

    file_info_mapping, metadata_mapping = load_archive_manifest(args.archive_manifest)
    print(f"Loaded {len(file_info_mapping)} file entries from manifest")

    process_pairs(args.root_dir, args.output_dir, file_info_mapping, metadata_mapping)

    print("Pair confirmation and reorganization completed successfully!")


if __name__ == '__main__':
    main()
