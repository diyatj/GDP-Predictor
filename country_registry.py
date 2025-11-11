from pathlib import Path
from typing import List, Optional

# Simple registry mapping country names to CSV filenames.
COUNTRY_CSV_MAP = {
    "United States": "datasets/US_GDP.csv",
}


def list_countries() -> List[str]:
    """Return the list of countries known to the registry (sorted).

    Returns an empty list if the registry is empty.
    """
    return sorted(COUNTRY_CSV_MAP.keys())


def get_csv_for_country(country: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    """Return a Path to the CSV file for the given country.

    - If country is None or 'All', returns None (meaning caller should use the default CSV).
    - If a mapping exists, returns a Path resolved relative to base_dir (or this file's parent).
    - If the mapped file does not exist, still returns the resolved Path (caller can check existence).
    """
    if country is None:
        return None
    if country == "All":
        return None

    fn = COUNTRY_CSV_MAP.get(country)
    if fn is None:
        return None

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    return (base_dir / fn).resolve()


def default_csv(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Return a best-effort default CSV path from the registry.

    If the registry has at least one mapping, returns the first file's path.
    Otherwise returns None.
    """
    if not COUNTRY_CSV_MAP:
        return None
    first_fn = next(iter(COUNTRY_CSV_MAP.values()))
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    return (base_dir / first_fn).resolve()
