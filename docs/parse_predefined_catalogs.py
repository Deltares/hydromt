import os
import numpy as np
import pandas as pd
from hydromt import DataCatalog
from pathlib import Path

FILE_ROOT = Path(__file__).parent

N_VERSIONS = 5  # number of versions to include in the dropdown

CATEGORIES = [
    "geography",
    "hydrography",
    "landuse",
    "hydro",
    "meteo",
    "ocean",
    "socio-economic",
    "topography",
    "climate",
    "other",
]

ATTRS = [
    "data_type",
    "driver",
    "version",
    "provider",
    "paper_ref",
    "paper_doi",
    "source_license",
    "source_url",
    "source_spatial_extent",
    "source_temporal_extent",
]


def write_panel(f, name: str, content: str="", level: int=0, item: str="dropdown") -> None:
    pad = "".ljust(level * 3)
    f.write(f"{pad}.. {item}:: {name}\n")
    f.write("\n")
    if content:
        pad = "".ljust((level + 1) * 3)
        for line in content.split("\n"):
            f.write(f"{pad}{line}\n")
        f.write("\n")


def write_nested_dropdown(name, df_dict: dict, note: str=""):
    path = Path(FILE_ROOT, f"_generated/{name.replace(' ', '_')}.rst")
    with open(path, mode="w") as f:
        write_panel(f, name, note, level=0)
        for i, version in enumerate(df_dict):
            df = df_dict[version]#.reset_index().set_index(["name", "provider", "version"])
            df = df.sort_index()
            name_str = f"{version}"
            if i == 0:
                name_str += " (latest)"
            write_panel(f, name_str, level=1)
            write_panel(f, "", level=2, item="tab-set")
            for category in CATEGORIES:
                if category == "other":
                    sources = df.index[~np.isin(df["category"], CATEGORIES)]
                else:
                    sources = df.index[df["category"] == category]
                if len(sources) > 0:
                    write_panel(f, category, level=3, item="tab-item")
                    write_sources_panel(f, df, level=4, sources=sources)


            write_panel(f, "all", level=3, item="tab-item")
            write_sources_panel(f, df, level=4)

    # return relative path
    return path.relative_to(FILE_ROOT)

def write_sources_panel(f, df, level, sources=None):
    attrs = [a for a in ATTRS if a in df.columns]  # accomodate older versions
    if sources is None:
        sources = df.index
    sources = sorted(set(sources))
    for source in sources:
        df0 = df.loc[source, attrs]
        if isinstance(df0, pd.Series):
            df0 = df.loc[[source], attrs]
        # combine all variants
        var_cols = [c for c in ['version', 'provider'] if df0[c].notna().any()]
        df_variants = df0[var_cols].sort_values(var_cols).reset_index(drop=True)
        variants: list[dict] = list(reversed(df_variants.to_dict(orient='index').values()))
        items = list(df0.iloc[0].drop(var_cols).items())
        if len(variants) > 0 and len(variants[0]) > 0: # list of dicts
            items += [('variants', variants)]
        # parse items to rst table
        summary = "\n".join(
            [parse_item(k, v) for k, v in items if not (v is np.nan or v is None)]
        )
        write_panel(f, source, summary, level=level)

def parse_item(k: str, v: str | list | dict) -> str:
    def _parse_dict(d: dict) -> str:
        return " ".join([f"**{k}:** {_parse(v)}" for k, v in d.items()])
    def _parse_list(l: list) -> str:
        return "\n".join([f"   - {_parse(d)}" for d in l])
    def _parse_str(s: str) -> str:
        if s.startswith("http"): # make hyperlink
            return f"`link <{s}>`__"
        else: # escape special characters
            return s.replace("*", "\\*").replace("_", "\\_")
    def _parse(v):
        if isinstance(v, dict):
            return _parse_dict(v)
        elif isinstance(v, list):
            return _parse_list(v)
        else:
            return _parse_str(str(v))
    k = k.replace("source_", "").replace("paper_", "")
    if k == "doi":
        v = f"`{v} <https://doi.org/{v}>`__"
    else:
        v = _parse(v)
    return f":{k}: {v}"

def write_predefined_catalogs_to_rst_panels(
    predefined_catalog_uri: Path = Path(FILE_ROOT, r"../data/predefined_catalogs.yml"),
    git_raw_uri: str = r"https://raw.githubusercontent.com/Deltares/hydromt",
) -> None:
    """Generate panels rst files from data catalogs to include in docs"""
    os.makedirs(Path(FILE_ROOT, "_generated"), exist_ok=True)
    data_cat = DataCatalog()
    data_cat.set_predefined_catalogs(predefined_catalog_uri)
    predefined_catalogs = data_cat.predefined_catalogs
    paths = []
    for name in predefined_catalogs:
        urlpath = predefined_catalogs[name].get("urlpath", "")
        note = predefined_catalogs[name].get("notes", "")
        df_dict = {}
        for iversion, version in enumerate(predefined_catalogs[name].get("versions", [])):
            if iversion >= N_VERSIONS:
                break
            githash = predefined_catalogs[name]['versions'][version]
            if urlpath.startswith(git_raw_uri) and githash == "main":
                # make sure to load the latest version from current branch
                local_path = Path(FILE_ROOT, urlpath.replace(f'{git_raw_uri}/{{version}}', '../'))
                data_cat.from_yml(local_path, catalog_name=name)
            else:
                try:
                    data_cat.from_predefined_catalogs(name, version=version)
                except OSError as e:
                    print(e)
                    continue
            df = data_cat.to_dataframe().sort_index().drop_duplicates("path")
            df_dict[version] = df.copy()
            data_cat._sources = {}  # reset
        path = write_nested_dropdown(name, df_dict, note=note)
        paths.append(path)
    with open(Path(FILE_ROOT, "_generated/predefined_catalogs.rst"), "w") as f:
        f.writelines(
            [f".. include:: ../{path}\n" for path in paths]
        )

if __name__ == "__main__":
    write_predefined_catalogs_to_rst_panels()
