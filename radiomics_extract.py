"""Extract PyRadiomics features from image and tumor-mask files."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
import yaml
from radiomics import featureextractor


def largest_section_mask(mask: sitk.Image) -> sitk.Image:
    """Retain only the axial section with the largest tumor area."""
    array = sitk.GetArrayFromImage(mask)
    if array.ndim != 3:
        raise ValueError("Tumor mask must be three-dimensional.")
    areas = (array > 0).reshape(array.shape[0], -1).sum(axis=1)
    if areas.max() == 0:
        raise ValueError("Tumor mask contains no foreground voxels.")
    selected = array.copy()
    selected[:] = 0
    z = int(areas.argmax())
    selected[z] = array[z] > 0
    result = sitk.GetImageFromArray(selected.astype("uint8"))
    result.CopyInformation(mask)
    return result


def main() -> None:
    installed_version = version("pyradiomics")
    if installed_version != "3.1.0":
        raise RuntimeError(
            "This analysis requires PyRadiomics 3.1.0; "
            f"found {installed_version}."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-suffix", default="_image.nii.gz")
    parser.add_argument("--mask-suffix", default="_mask.nii.gz")
    args = parser.parse_args()

    config = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    )
    output_path = Path(
        args.output or config["data"]["radiomics_table"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params_file = config.get("radiomics", {}).get("params_file")
    if params_file:
        extractor = featureextractor.RadiomicsFeatureExtractor(
            params_file
        )
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings["force2D"] = True
    extractor.settings["force2Ddimension"] = 0

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    rows = []

    for image_path in sorted(
        image_dir.glob(f"*{args.image_suffix}")
    ):
        patient_id = image_path.name[: -len(args.image_suffix)]
        mask_path = mask_dir / f"{patient_id}{args.mask_suffix}"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing mask for {patient_id}: {mask_path}"
            )

        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        if image.GetDimension() != 3 or mask.GetDimension() != 3:
            raise ValueError(
                f"Image and mask for {patient_id} must be three-dimensional."
            )
        if image.GetSize() != mask.GetSize():
            raise ValueError(
                f"Image and mask sizes differ for {patient_id}: "
                f"{image.GetSize()} vs {mask.GetSize()}."
            )
        result = extractor.execute(image, largest_section_mask(mask))
        row = {"patient_id": patient_id}
        row.update(
            {
                key: value
                for key, value in result.items()
                if not key.startswith("diagnostics_")
            }
        )
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No image files matched suffix: {args.image_suffix}"
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} patients to {output_path}")


if __name__ == "__main__":
    main()
