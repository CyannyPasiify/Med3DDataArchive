# -*- coding: utf-8 -*-
"""
Data Pair Confirmation and Reorganization Script for Duke-Breast-FGT-Segmentation Dataset

This script confirms and reorganizes image-mask pairs from the Duke-Breast-FGT-Segmentation dataset,
ensuring metadata consistency, remapping mask labels according to provided label maps, and organizing
files into a standardized directory structure grouped by subset and patient ID.

Parameters:
    -r, --root_dir: Root directory of the dataset containing Images and Masks directories
    -o, --output_dir: Output directory where reorganized data will be saved
    -am, --archive_manifest: Path to Excel manifest file containing dataset metadata
    -e, --label_explanation: Path to label_map.yaml file containing label mapping information
    -s, --sheet_name: Optional sheet name to rename worksheet to

Usage Examples:
    python 02_confirm_pairs.py -r /path/to/Duke-Breast -o /path/to/output -am /path/to/manifest.xlsx -e /path/to/label_map.yaml
    python 02_confirm_pairs.py --root_dir /path/to/Duke-Breast --output_dir /path/to/output --archive_manifest /path/to/manifest.xlsx --label_explanation /path/to/label_map.yaml --sheet_name "Processed Data"
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage
from typing import Dict, List, Tuple, Any, Optional, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, output_dir, archive_manifest, label_explanation, and sheet_name
    """
    parser = argparse.ArgumentParser(
        description='Confirm and reorganize image-mask pairs for Duke-Breast-FGT-Segmentation dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/Duke-Breast -o /path/to/output -am /path/to/manifest.xlsx -e /path/to/label_map.yaml
  %(prog)s --root_dir /path/to/Duke-Breast --output_dir /path/to/output --archive_manifest /path/to/manifest.xlsx --label_explanation /path/to/label_map.yaml --sheet_name "Processed Data"
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing Images and Masks directories'
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

    parser.add_argument(
        '-e', '--label_explanation',
        type=str,
        required=True,
        help='Path to label_map.yaml file containing label mapping information'
    )

    parser.add_argument(
        '-s', '--sheet_name',
        type=str,
        default=None,
        help='Optional sheet name to rename worksheet to'
    )

    return parser.parse_args()


def load_label_map(label_map_path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load label map from YAML file.
    
    Args:
        label_map_path (Union[str, Path]): Path to label_map.yaml file
        
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: (full_form_label_map, short_form_index_map)
    """
    label_map_file = Path(label_map_path)
    
    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map file not found: {label_map_path}")
    
    with open(label_map_file, 'r', encoding='utf-8') as f:
        label_map = yaml.safe_load(f)
    
    full_form_label_map = label_map.get('full_form_label_map', {})
    full_form_label_map.update({'breast': full_form_label_map.pop('breast residue')})
    short_form_index_map = label_map.get('short_form_index_map', {})
    
    return full_form_label_map, short_form_index_map


def parse_seg_type(seg_type: str) -> Dict[int, str]:
    """
    Parse segmentation type string into a dictionary of original label index to label name.
    
    Args:
        seg_type (str): Segmentation type string in format ({label_index_origin}_{label_name_origin})+
        
    Returns:
        Dict[int, str]: Dictionary mapping original label indices to label names (lowercase with spaces)
    """
    label_map = {}
    segments = seg_type.split('_')
    
    i = 0
    while i < len(segments):
        if segments[i].isdigit():
            label_index = int(segments[i])
            label_name_parts = []
            i += 1
            
            while i < len(segments) and not segments[i].isdigit():
                label_name_parts.append(segments[i])
                i += 1
            
            if label_name_parts:
                label_name = '_'.join(label_name_parts)
                # Convert to lowercase and replace underscores with spaces
                label_name = label_name.lower().replace('_', ' ')
                label_map[label_index] = label_name
        else:
            i += 1
    
    return label_map


def create_label_remap(original_label_map: Dict[int, str], full_form_label_map: Dict[str, int]) -> Dict[int, int]:
    """
    Create label remapping from original indices to new indices based on full_form_label_map.
    
    Args:
        original_label_map (Dict[int, str]): Original label index to label name mapping
        full_form_label_map (Dict[str, int]): Full form label map from YAML file
        
    Returns:
        Dict[int, int]: Label remapping dictionary
    """
    remap = {}
    
    for orig_idx, orig_name in original_label_map.items():
        # Special handling for Breast_Remained
        if 'breast remained' in orig_name:
            # Remove 'remained' suffix and try to match
            base_name = 'breast'
            if base_name in full_form_label_map:
                remap[orig_idx] = full_form_label_map[base_name]
        else:
            # Direct matching
            if orig_name in full_form_label_map:
                remap[orig_idx] = full_form_label_map[orig_name]
    
    return remap


def remap_mask_labels(mask_data: np.ndarray, remap_dict: Dict[int, int]) -> np.ndarray:
    """
    Remap labels in mask data according to remap dictionary.
    
    Args:
        mask_data (np.ndarray): Original mask data
        remap_dict (Dict[int, int]): Label remapping dictionary
        
    Returns:
        np.ndarray: Mask data with remapped labels
    """
    remapped = np.zeros_like(mask_data, dtype=np.uint8)
    
    for orig_idx, new_idx in remap_dict.items():
        remapped[mask_data == orig_idx] = new_idx
    
    return remapped


def load_archive_manifest(manifest_path: Union[str, Path]) -> Tuple[
    Dict[str, Dict[str, Optional[str]]], Dict[str, Dict[str, Any]]
]:
    """
    Load archive manifest Excel file and create file information mapping.
    
    Args:
        manifest_path (Union[str, Path]): Path to Excel manifest file
        
    Returns:
        Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, Dict[str, Any]]]: 
            - file_info_mapping: Dictionary mapping pid to file information
            - metadata_mapping: Dictionary mapping pid to sample metadata
    """
    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        raise FileNotFoundError(f"Archive manifest file not found: {manifest_path}")

    df = pd.read_excel(manifest_file, engine='openpyxl', dtype={'pid': str})

    required_columns = ['file_path', 'pid', 'type', 'subset', 'primary']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in archive manifest: {manifest_path}")

    file_info_mapping: Dict[str, Dict[str, Optional[str]]] = {}
    metadata_mapping: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        file_path = str(row['file_path'])
        pid = str(row['pid'])
        file_type = str(row['type']).lower() if pd.notna(row['type']) else ''
        subset = str(row['subset']) if pd.notna(row['subset']) else ''
        primary = int(row['primary']) if pd.notna(row['primary']) else 2

        if pid not in file_info_mapping:
            file_info_mapping[pid] = {
                'primary_image': None,
                'mask_1_breast': None,
                'mask_combined': None,
                'subset': subset
            }

        # Store metadata for primary images
        if file_type == 'v3d' and primary == 1:
            file_info_mapping[pid]['primary_image'] = file_path
            metadata_mapping[pid] = {
                'pid': pid,
                'subset': subset
            }
        elif file_type == 'm3d':
            # Extract seg_type from filename
            # Filename format: Segmentation_{pid}_{seg_type}.nii.gz
            # Since pid may contain underscores, we need to use the known pid to extract seg_type
            file_name = Path(file_path).name
            if file_name.startswith('Segmentation_'):
                # Remove 'Segmentation_' prefix
                remaining = file_name[len('Segmentation_'):]
                # Remove '.nii.gz' suffix
                remaining = remaining.replace('.nii.gz', '')
                # Remove the pid prefix to get seg_type
                if remaining.startswith(pid):
                    seg_type = remaining[len(pid):].lstrip('_')
                    if seg_type == '1_Breast':
                        file_info_mapping[pid]['mask_1_breast'] = file_path
                    elif seg_type == '1_Breast_Remained_2_Fibroglandular_Tissue_3_Blood_Vessel':
                        file_info_mapping[pid]['mask_combined'] = file_path

    return file_info_mapping, metadata_mapping


def check_metadata_consistency(image_meta: Dict[str, Any], mask_meta: Dict[str, Any]) -> bool:
    """
    Check if image and mask metadata (spatial_shape and affine) are consistent.
    
    Args:
        image_meta (Dict[str, Any]): Image metadata dictionary
        mask_meta (Dict[str, Any]): Mask metadata dictionary
        
    Returns:
        bool: True if metadata is consistent, False otherwise
    """
    if 'spatial_shape' not in image_meta or 'spatial_shape' not in mask_meta:
        return False
    
    if 'affine' not in image_meta or 'affine' not in mask_meta:
        return False
    
    image_shape = image_meta['spatial_shape']
    mask_shape = mask_meta['spatial_shape']

    if not (image_shape == mask_shape).all():
        return False

    image_affine = image_meta['affine']
    mask_affine = mask_meta['affine']

    if not (image_affine == mask_affine).all():
        return False

    return True


def process_pairs(root_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  archive_manifest: Union[str, Path],
                  label_explanation: Union[str, Path],
                  sheet_name: Optional[str] = None) -> None:
    """
    Process all image-mask pairs and reorganize them into output directory.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing Images and Masks directories
        output_dir (Union[str, Path]): Output directory for reorganized data
        archive_manifest (Union[str, Path]): Path to Excel manifest file
        label_explanation (Union[str, Path]): Path to label_map.yaml file
        sheet_name (Optional[str]): Optional sheet name to rename worksheet to
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    # Load label maps
    full_form_label_map, short_form_index_map = load_label_map(label_explanation)
    print(f"Loaded label maps with {len(full_form_label_map)} full form labels")

    # Load archive manifest
    file_info_mapping, metadata_mapping = load_archive_manifest(archive_manifest)
    print(f"Loaded {len(file_info_mapping)} sample entries from manifest")

    # Filter samples that have primary image and both required masks
    valid_samples = [pid for pid, info in file_info_mapping.items() 
                    if info['primary_image'] and info['mask_1_breast'] and info['mask_combined']]
    print(f"Found {len(valid_samples)} valid samples with primary image and both required masks")

    # Process each valid sample
    total_processed = 0
    total_issues = 0

    loader: LoadImage = LoadImage(image_only=False, dtype=None)

    with tqdm(total=len(valid_samples), desc='Processing samples') as pbar:
        for pid in valid_samples:
            pbar.set_description(f'Processing {pid}')
            
            info = file_info_mapping[pid]
            subset = info['subset']
            primary_image_path = root_path / info['primary_image']
            mask_1_breast_path = root_path / info['mask_1_breast']
            mask_combined_path = root_path / info['mask_combined']

            # Create output directory structure
            subset_dir = output_path / subset
            sample_dir = subset_dir / pid
            sample_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Load primary image
                image_data, image_meta = loader(str(primary_image_path))

                # Process 1_Breast mask
                mask_1_data, mask_1_meta = loader(str(mask_1_breast_path))
                
                # Parse seg_type and create label remap
                # Filename format: Segmentation_{pid}_{seg_type}.nii.gz
                # Since pid may contain underscores, we need to use the known pid to extract seg_type
                mask_1_filename = Path(mask_1_breast_path).name
                if mask_1_filename.startswith('Segmentation_'):
                    remaining = mask_1_filename[len('Segmentation_'):]
                    remaining = remaining.replace('.nii.gz', '')
                    if remaining.startswith(pid):
                        seg_type_1 = remaining[len(pid):].lstrip('_')
                        orig_label_map_1 = parse_seg_type(seg_type_1)
                        label_remap_1 = create_label_remap(orig_label_map_1, full_form_label_map)
                
                # Remap labels
                remapped_mask_1 = remap_mask_labels(mask_1_data, label_remap_1)

                # Process combined mask
                mask_combined_data, mask_combined_meta = loader(str(mask_combined_path))
                
                # Parse seg_type and create label remap
                # Filename format: Segmentation_{pid}_{seg_type}.nii.gz
                mask_combined_filename = Path(mask_combined_path).name
                if mask_combined_filename.startswith('Segmentation_'):
                    remaining = mask_combined_filename[len('Segmentation_'):]
                    remaining = remaining.replace('.nii.gz', '')
                    if remaining.startswith(pid):
                        seg_type_combined = remaining[len(pid):].lstrip('_')
                        orig_label_map_combined = parse_seg_type(seg_type_combined)
                        label_remap_combined = create_label_remap(orig_label_map_combined, full_form_label_map)
                
                # Remap labels
                remapped_mask_combined = remap_mask_labels(mask_combined_data, label_remap_combined)

                # Check metadata consistency for masks
                if not check_metadata_consistency(image_meta, mask_1_meta):
                    print(f"Warning: Metadata mismatch for {pid} - 1_Breast mask")
                    print(f"  Image shape: {image_meta['spatial_shape']}, Mask shape: {mask_1_meta['spatial_shape']}")
                    print(f"  Copying image metadata to mask...")
                    mask_1_meta = image_meta.copy()
                    total_issues += 1

                if not check_metadata_consistency(image_meta, mask_combined_meta):
                    print(f"Warning: Metadata mismatch for {pid} - combined mask")
                    print(f"  Image shape: {image_meta['spatial_shape']}, Mask shape: {mask_combined_meta['spatial_shape']}")
                    print(f"  Copying image metadata to mask...")
                    mask_combined_meta = image_meta.copy()
                    total_issues += 1

                # Save primary image as volume
                volume_path = sample_dir / f'{pid}_volume.nii.gz'
                saver_image = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.float32)
                saver_image(image_data, meta_data=image_meta, filename=str(volume_path).replace('.nii.gz', ''))

                # Save 1_Breast mask as mask_mass
                mask_mass_path = sample_dir / f'{pid}_mask_mass.nii.gz'
                saver_mask_1 = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)
                saver_mask_1(remapped_mask_1, meta_data=mask_1_meta, filename=str(mask_mass_path).replace('.nii.gz', ''))

                # Save combined mask as mask
                mask_path = sample_dir / f'{pid}_mask.nii.gz'
                saver_mask_combined = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)
                saver_mask_combined(remapped_mask_combined, meta_data=mask_combined_meta, filename=str(mask_path).replace('.nii.gz', ''))

                # Save info.yaml
                info_yaml_path = sample_dir / f'{pid}_info.yaml'
                with open(info_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(metadata_mapping[pid], f, default_flow_style=False, allow_unicode=True)

                total_processed += 1

            except Exception as e:
                print(f"Error processing {pid}: {str(e)}")
                total_issues += 1

            pbar.update(1)

    print(f"\nProcessing completed!")
    print(f"Total samples processed: {total_processed}")
    print(f"Total issues encountered: {total_issues}")


def main() -> None:
    """
    Main function to orchestrate pair confirmation and reorganization process.
    """
    args = parse_args()

    print(f"Processing dataset from: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Archive manifest: {args.archive_manifest}")
    print(f"Label explanation: {args.label_explanation}")
    if args.sheet_name:
        print(f"Sheet name: {args.sheet_name}")

    process_pairs(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        archive_manifest=args.archive_manifest,
        label_explanation=args.label_explanation,
        sheet_name=args.sheet_name
    )

    print("Pair confirmation and reorganization completed successfully!")


if __name__ == '__main__':
    main()
