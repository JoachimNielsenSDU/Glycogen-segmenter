from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


PIXEL_SIZE_NM = 0.7698
THICKNESS_UM = 0.06

INPUT_CANDIDATES = [
    Path("Fasting_imf_output") / "glycogen_distribution_combined_threshold_0p85.csv",
    Path("fasting_imf_output") / "glycogen_distribution_combined_threshold_0p85.csv",
]

OUTPUT_XLSX = Path("Fasting_imf_output") / "glycogen_distribution_stereology_threshold_0p85_adj.xlsx"


def _normalize_region_name(region: str) -> str:
    return "".join(ch.lower() for ch in str(region) if ch.isalnum())


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def _compute_region_set(
    glycogen_area: float,
    mean_feret_diameter: float,
    total_area: float,
    t_um: float,
    aa_slope: float | None = None,
    aa_intercept: float | None = None,
) -> dict[str, float]:
    aa = _safe_div(glycogen_area, total_area)
    if (
        np.isfinite(aa)
        and aa_slope is not None
        and aa_intercept is not None
        and np.isfinite(aa_slope)
        and np.isfinite(aa_intercept)
    ):
        aa = aa - ((aa_slope * aa) + aa_intercept)

    h = (mean_feret_diameter * PIXEL_SIZE_NM) / 1000.0
    s = math.pi * (h**2)
    na = _safe_div(aa, math.pi * ((0.5 * h) ** 2))
    nv = _safe_div(na, t_um + h)
    sv = nv * s if np.isfinite(nv) and np.isfinite(s) else np.nan

    ba = np.nan
    if np.isfinite(sv) and np.isfinite(nv) and np.isfinite(h):
        ba = ((math.pi * sv) / 4.0) + (t_um * nv * math.pi * h)

    vv = np.nan
    if np.isfinite(aa) and np.isfinite(ba) and np.isfinite(na) and np.isfinite(h):
        vv = 100.0 * (
            aa - (t_um * ((ba / math.pi) - (_safe_div(na * t_um * h, (t_um + h)))))
        )

    numerical_density = np.nan
    if np.isfinite(vv) and np.isfinite(h):
        numerical_density = _safe_div(vv/100, (3.0 / 4.0) * math.pi * ((h / 2.0) ** 3))

    return {
        "aa": aa,
        "h": h,
        "s": s,
        "na": na,
        "nv": nv,
        "sv": sv,
        "ba": ba,
        "vv_pct": vv,
        "numerical_density": numerical_density,
    }


def _pick_input_file() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find glycogen_distribution_combined_threshold_0p85.csv in "
        "Fasting_imf_output or fasting_imf_output."
    )


def main() -> None:
    input_csv = _pick_input_file()

    df = pd.read_csv(input_csv)
    required_cols = {
        "Subfolder",
        "Image",
        "Region",
        "Area",
        "Glycogen Area",
        "Mean Feret Diameter",
        "Z-disc max feret",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = ["Area", "Glycogen Area", "Mean Feret Diameter", "Z-disc max feret"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["_region"] = df["Region"].map(_normalize_region_name)

    wanted_regions = ["intermyofibrillar", "mitochondria", "zdisc", "intra"]

    rows: list[dict[str, float | str]] = []

    for (subfolder, image), grp in df.groupby(["Subfolder", "Image"], sort=True):
        region_map = {
            region: grp.loc[grp["_region"] == region].iloc[0]
            for region in wanted_regions
            if not grp.loc[grp["_region"] == region].empty
        }

        total_area = float(
            sum(float(region_map[r]["Area"]) for r in wanted_regions if r in region_map)
        )

        out_row: dict[str, float | str] = {
            "Subfolder": subfolder,
            "Image": image,
            "sum_area_imf_mito_zdisc_intra": total_area,
        }

        if "intermyofibrillar" in region_map:
            imf = _compute_region_set(
                glycogen_area=float(region_map["intermyofibrillar"]["Glycogen Area"]),
                mean_feret_diameter=float(region_map["intermyofibrillar"]["Mean Feret Diameter"]),
                total_area=total_area,
                t_um=THICKNESS_UM,
                aa_slope=-0.1767,
                aa_intercept=0.00299,
            )
            out_row.update({f"IMF_{k}": v for k, v in imf.items()})
        else:
            out_row.update({f"IMF_{k}": np.nan for k in ["aa", "h", "s", "na", "nv", "sv", "ba", "vv_pct", "numerical_density"]})

        if "intra" in region_map:
            intra = _compute_region_set(
                glycogen_area=float(region_map["intra"]["Glycogen Area"]),
                mean_feret_diameter=float(region_map["intra"]["Mean Feret Diameter"]),
                total_area=total_area,
                t_um=THICKNESS_UM,
                aa_slope=-0.1561,
                aa_intercept=0.001728,
            )
            out_row.update({f"intra_{k}": v for k, v in intra.items()})
        else:
            out_row.update({f"intra_{k}": np.nan for k in ["aa", "h", "s", "na", "nv", "sv", "ba", "vv_pct", "vv_per_particle_volume"]})

        intra_area = float(region_map["intra"]["Area"]) if "intra" in region_map else np.nan
        zdisc_area = float(region_map["zdisc"]["Area"]) if "zdisc" in region_map else np.nan
        out_row["intra_zdisc_area_per_total_area"] = _safe_div(
            sum(area for area in [intra_area, zdisc_area] if np.isfinite(area)),
            total_area,
        )
        out_row["intra_vv_pct_per_intra_zdisc_area"] = _safe_div(
            out_row.get("intra_vv_pct", np.nan),
            out_row["intra_zdisc_area_per_total_area"],
        )

        mito_area = float(region_map["mitochondria"]["Area"]) if "mitochondria" in region_map else np.nan
        out_row["mito_vv_pct"] = 100.0 * _safe_div(mito_area, total_area)

        if "zdisc" in region_map:
            zdisc_max_feret = float(region_map["zdisc"]["Z-disc max feret"])
            out_row["zdiscwidth"] = _safe_div(zdisc_area, zdisc_max_feret) * PIXEL_SIZE_NM
        else:
            out_row["zdiscwidth"] = np.nan

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["Subfolder", "Image"]).reset_index(drop=True)

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_XLSX
    try:
        out_df.to_excel(output_path, index=False)
    except PermissionError:
        output_path = OUTPUT_XLSX.with_name(f"{OUTPUT_XLSX.stem}_v2{OUTPUT_XLSX.suffix}")
        out_df.to_excel(output_path, index=False)

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
