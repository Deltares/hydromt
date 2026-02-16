from pathlib import Path

import pytest
import xarray as xr

from hydromt._utils.naming_convention import expand_uri_paths


def create_nc(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"x": 1}).to_netcdf(path)
    return path


def test_expand_uri_paths_wildcard_only(tmp_path: Path):
    paths = [create_nc(tmp_path / f"file{i}.nc") for i in range(3)]
    results = expand_uri_paths(str(tmp_path / "*.nc"))
    assert results == {path.stem: str(path) for path in paths}


def test_expand_uri_paths_single_placeholder(tmp_path: Path):
    f1 = create_nc(tmp_path / "forcing_2020.nc")
    f2 = create_nc(tmp_path / "forcing_2021.nc")
    uri = str(tmp_path / "forcing_{year}.nc")
    results = expand_uri_paths(uri, placeholders=["year"])
    assert results == {"2020": str(f1), "2021": str(f2)}


def test_expand_uri_paths_multiple_placeholders(tmp_path: Path):
    f1 = create_nc(tmp_path / "forcing_2020_temp.nc")
    f2 = create_nc(tmp_path / "forcing_2021_precip.nc")
    uri = str(tmp_path / "forcing_{year}_{variable}.nc")
    results = expand_uri_paths(uri, placeholders=["year", "variable"])
    assert results == {"2020.temp": str(f1), "2021.precip": str(f2)}


def test_expand_uri_paths_wildcard_plus_placeholders(tmp_path: Path):
    f1 = create_nc(tmp_path / "forcing_one_2020_temp.nc")
    f2 = create_nc(tmp_path / "forcing_two_2021_precip.nc")
    uri = str(tmp_path / "forcing_*_{year}_{variable}.nc")
    results = expand_uri_paths(uri, placeholders=["year", "variable"])
    assert results == {"2020.temp": str(f1), "2021.precip": str(f2)}


def test_expand_uri_paths_placeholders_in_dir(tmp_path: Path):
    f1 = create_nc(tmp_path / "2020" / "temp" / "forcing.nc")
    f2 = create_nc(tmp_path / "2021" / "precip" / "forcing.nc")
    uri = str(tmp_path / "{year}/{variable}/forcing.nc")
    results = expand_uri_paths(uri, placeholders=["year", "variable"])
    assert results == {"2020.temp": str(f1), "2021.precip": str(f2)}


def test_expand_uri_paths_wildcard_and_placeholder_in_dir(tmp_path: Path):
    f1 = create_nc(tmp_path / "runA/2020/temp/forcing.nc")
    f2 = create_nc(tmp_path / "runB/2021/precip/forcing.nc")
    _ = create_nc(tmp_path / "runB/2021/precip/should_not_match.nc")

    uri = str(tmp_path / "*/{year}/{variable}/forcing.nc")
    results = expand_uri_paths(uri, placeholders=["year", "variable"])
    assert results == {"2020.temp": str(f1), "2021.precip": str(f2)}


def test_expand_uri_paths_wildcard_and_placeholder_in_dir_(tmp_path: Path):
    f1 = create_nc(tmp_path / "2020" / "temp" / "precip_forcing.nc")
    f2 = create_nc(tmp_path / "2020" / "press" / "press_forcing.nc")
    f3 = create_nc(tmp_path / "2021" / "temp" / "precip_forcing.nc")
    f4 = create_nc(tmp_path / "2022" / "temp" / "precip_forcing.nc")

    uri = str(tmp_path / "{year}/{variable}/*_forcing.nc")
    results = expand_uri_paths(uri, placeholders=["year", "variable"])
    assert results == {
        "2020.temp": str(f1),
        "2020.press": str(f2),
        "2021.temp": str(f3),
        "2022.temp": str(f4),
    }


def test_expand_uri_paths_wildcard_and_placeholder_not_unique_names_raises(
    tmp_path: Path,
):
    root = tmp_path / "2020/temp"
    root.mkdir(parents=True)
    create_nc(root / "some_forcing.nc")
    create_nc(root / "other_forcing.nc")

    uri = str(tmp_path / "{year}/{variable}/*_forcing.nc")
    with pytest.raises(ValueError, match=r"Duplicate dataset name"):
        expand_uri_paths(uri, placeholders=["year", "variable"])
