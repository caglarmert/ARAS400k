import os
import re
import rasterio
from rasterio.windows import from_bounds, Window
import numpy as np
from pathlib import Path
from rasterio.enums import Resampling
from PIL import Image

def extract_and_convert_to_png(
    worldcover_dir,
    s2rgb_dir,
    output_dir,
    patch_size=256,
    max_patches=None,
    skip_all_water=True,
    verbose=True
):
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Class remapping
    remap = {
        90: 30,   # Wetland → Grassland
        100: 30,  # Moss → Grassland
        95: 10,   # Mangrove → Tree cover
        70: 60,   # Snow → Barren
        # Keep others unchanged (including 0 = No data)
    }
    # Color map after remapping
    wc_colors = {
        10: [0, 100, 0],        # Tree cover
        20: [255, 182, 193],    # Shrubland
        30: [154, 205, 50],     # Grassland (includes wetland & moss)
        40: [255, 215, 0],      # Cropland
        50: [139, 69, 19],      # Built-up
        60: [211, 211, 211],    # Barren/sparse (includes snow)
        80: [0, 0, 255],        # Water
        0: [0, 0, 0]            # No data (will be filtered out)
    }


    max_class = max(wc_colors.keys())
    color_array = np.zeros((max_class + 1, 3), dtype=np.uint8)
    for cls, color in wc_colors.items():
        color_array[cls] = color

    # Find input files
    wc_files = {}
    for root, _, files in os.walk(worldcover_dir):
        for f in files:
            if f.endswith('_Map.tif'):
                match = re.search(r'_([NS]\d{2}[EW]\d{3})_Map\.tif$', f)
                if match:
                    tile_id = match.group(1)
                    wc_files[tile_id] = os.path.join(root, f)

    s2_files = {}
    for root, _, files in os.walk(s2rgb_dir):
        for f in files:
            if f.endswith('_S2RGBNIR.tif'):
                match = re.search(r'_([NS]\d{2}[EW]\d{3})_S2RGBNIR\.tif$', f)
                if match:
                    tile_id = match.group(1)
                    s2_files[tile_id] = os.path.join(root, f)

    if verbose:
        print(f"Found {len(wc_files)} WorldCover tiles and {len(s2_files)} S2 tiles.")

    patch_count = 0
    skipped_black_s2 = 0
    skipped_missing_wc = 0
    skipped_no_data = 0
    skipped_water = 0

    for s2_tile_id, s2_path in sorted(s2_files.items()):
        if max_patches is not None and patch_count >= max_patches:
            break

        lat = int(s2_tile_id[1:3])
        lon = int(s2_tile_id[4:7])
        hemi_ns = s2_tile_id[0]
        hemi_ew = s2_tile_id[3]

        if hemi_ns == 'S':
            lat = -lat
        if hemi_ew == 'W':
            lon = -lon

        candidates = []
        for dlat in [-1, 0, 1]:
            for dlon in [-1, 0, 1]:
                nlat = lat + dlat
                nlon = lon + dlon
                n_hemi_ns = 'N' if nlat >= 0 else 'S'
                n_hemi_ew = 'E' if nlon >= 0 else 'W'
                nlat_str = f"{abs(nlat):02d}"
                nlon_str = f"{abs(nlon):03d}"
                wc_candidate = f"{n_hemi_ns}{nlat_str}{n_hemi_ew}{nlon_str}"
                if wc_candidate in wc_files:
                    candidates.append(wc_candidate)

        if not candidates:
            if verbose:
                print(f"No WorldCover tile for S2 tile {s2_tile_id}")
            continue

        try:
            with rasterio.open(s2_path) as s2_src:
                s2_height, s2_width = s2_src.shape
                s2_transform = s2_src.transform

                n_patches_x = s2_width // patch_size
                n_patches_y = s2_height // patch_size
                if n_patches_x == 0 or n_patches_y == 0:
                    continue

                for i in range(n_patches_y):
                    if max_patches is not None and patch_count >= max_patches:
                        break
                    for j in range(n_patches_x):
                        if max_patches is not None and patch_count >= max_patches:
                            break

                        col_off = j * patch_size
                        row_off = i * patch_size
                        left, top = s2_transform * (col_off, row_off)
                        right, bottom = s2_transform * (col_off + patch_size, row_off + patch_size)

                        # Read S2 RGB
                        s2_window = Window(col_off, row_off, patch_size, patch_size)
                        s2_patch = s2_src.read([1, 2, 3], window=s2_window)
                        s2_patch = np.transpose(s2_patch, (1, 2, 0))

                        if np.all(s2_patch == 0):
                            skipped_black_s2 += 1
                            continue

                        # Read WC with clipped window
                        wc_patch = None
                        for wc_tile_id in candidates:
                            wc_path = wc_files[wc_tile_id]
                            try:
                                with rasterio.open(wc_path) as wc_src:
                                    wc_window = from_bounds(left, bottom, right, top, wc_src.transform)
                                    wc_window = wc_window.intersection(
                                        Window(0, 0, wc_src.width, wc_src.height)
                                    )
                                    if wc_window.width <= 0 or wc_window.height <= 0:
                                        continue

                                    wc_data = wc_src.read(
                                        1,
                                        window=wc_window,
                                        out_shape=(patch_size, patch_size),
                                        resampling=Resampling.nearest
                                    )
                                    wc_patch = wc_data
                                    break
                            except Exception:
                                continue

                        if wc_patch is None:
                            skipped_missing_wc += 1
                            continue

                        # Apply remapping
                        wc_patch_mapped = wc_patch.copy()
                        for old_cls, new_cls in remap.items():
                            wc_patch_mapped[wc_patch == old_cls] = new_cls

                        # === Skip if any no data class (0) ===
                        if np.any(wc_patch_mapped == 0):
                            skipped_no_data += 1
                            continue

                        # === Skip if 90% water (class 80) ===
                        if skip_all_water:
                            water_ratio = np.mean(wc_patch_mapped == 80)
                            if water_ratio > 0.9:
                                skipped_water += 1
                                continue

                        patch_name = f"{s2_tile_id}_{i:03d}_{j:03d}"

                        s2_png = (s2_patch / s2_patch.max() * 255).astype(np.uint8) if s2_patch.max() > 0 else s2_patch.astype(np.uint8)

                        mask_png = color_array[wc_patch_mapped]

                        Image.fromarray(s2_png).save(images_dir / f"{patch_name}.png")
                        Image.fromarray(mask_png).save(masks_dir / f"{patch_name}.png")

                        patch_count += 1
                        if verbose and patch_count % 100 == 0:
                            print(f"Processed {patch_count} patches...")

        except Exception as e:
            if verbose:
                print(f"Error processing S2 tile {s2_tile_id}: {e}")
            continue

    if verbose:
        print(f"Done! Total patches: {patch_count}")
        print(f"Skipped (black S2 image): {skipped_black_s2}")
        print(f"Skipped (missing WorldCover data): {skipped_missing_wc}")
        print(f"Skipped (contains no data class 0): {skipped_no_data}")
        print(f"Skipped (water >90%): {skipped_water}")
        print(f"Images: {images_dir}")
        print(f"Masks:  {masks_dir}")

# === CONFIGURE ===
if __name__ == "__main__":
    worldcover_dir = "worldcover_2021"
    s2rgb_dir = "S2RGB_2021"
    output_dir = "ARAS"
    patch_size = 256
    max_patches = None
    skip_all_water = True
    verbose = True

    extract_and_convert_to_png(
        worldcover_dir=worldcover_dir,
        s2rgb_dir=s2rgb_dir,
        output_dir=output_dir,
        patch_size=patch_size,
        max_patches=max_patches,
        skip_all_water=skip_all_water,
        verbose=verbose
    )