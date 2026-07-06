from pathlib import Path
from urllib.request import urlretrieve


DOIS = {
    "2020-01": "925117",
    "2020-02": "925119",
    "2020-03": "925120",
    "2020-04": "925121",
    "2020-05": "925122",
    "2020-06": "925123",
    "2020-07": "925124",
    "2020-08": "925125",
    "2020-09": "925126",
    "2020-10": "925112",
    "2020-11": "926549",
    "2020-12": "926556",
}


def main() -> None:
    out_dir = Path(r"E:\solarfusionnet_preprocess\raw_station_2020")
    out_dir.mkdir(parents=True, exist_ok=True)
    for month, doi_id in DOIS.items():
        out = out_dir / f"IZA_radiation_{month}.tab"
        if out.exists() and out.stat().st_size > 100_000:
            print(f"exists {out}")
            continue
        url = f"https://doi.pangaea.de/10.1594/PANGAEA.{doi_id}?format=textfile"
        print(f"download {month} {url}")
        urlretrieve(url, out)
        print(f"wrote {out} {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
