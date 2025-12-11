from pathlib import Path
from typing import List, Optional

# Simple registry mapping country names to CSV filenames.
COUNTRY_CSV_MAP = {
    "United States": "datasets/US_GDP.csv",
    "Japan": "datasets/Japan_GDP.csv",
    "Israel": "datasets/Israel_GDP.csv",
}

# Optional: registry for country-specific events (shock factors)
COUNTRY_EVENTS_CSV_MAP = {
    "United States": "datasets/us_events.csv",
    "Japan": "datasets/japan_events.csv",
    "Israel": "datasets/israel_events.csv",
}

#registry for prediction files
COUNTRY_PREDICT_MAP = {
    "United States": "datasets/US_predict.csv",
    "Japan": "datasets/Japan_predict.csv",
    "Israel": "datasets/Israel_predict.csv",
}


def list_countries() -> List[str]:
    return sorted(COUNTRY_CSV_MAP.keys())

#helper function to get csv files
def get_csv(country: Optional[str], mapping: dict, base_dir: Optional[Path] = None) -> Optional[Path]:
    if not country or country == "All":
        return None

    fn = mapping.get(country)
    if fn is None:
        return None

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    return (base_dir / fn).resolve()

#getting csv file with actual GDP values
def get_csv_for_country(country: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    return get_csv(country, COUNTRY_CSV_MAP, base_dir)

#getting csv file with shock events
def get_events_csv_for_country(country: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    return get_csv(country, COUNTRY_EVENTS_CSV_MAP, base_dir)

#getting csv file with GDP prediction
def get_predict_csv_for_country(country: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    return get_csv(country, COUNTRY_PREDICT_MAP, base_dir)

#default csv
def default_csv(base_dir: Optional[Path] = None) -> Optional[Path]:
    if not COUNTRY_CSV_MAP:
        return None
    first_fn = next(iter(COUNTRY_CSV_MAP.values()))
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    return (base_dir / first_fn).resolve()