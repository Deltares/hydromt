from pathlib import Path

import xarray as xr

from hydromt._utils.naming_convention import expand_uri_paths


def create_nc(path: Path, value=1):
    xr.Dataset({"x": value}).to_netcdf(path)
    return path


def test_expand_uri_paths_wildcard_only(tmp_path):
    paths = [create_nc(tmp_path / f"file{i}.nc") for i in range(3)]
    results = {name: path for path, name in expand_uri_paths(str(tmp_path / "*.nc"))}
    assert results == {path.stem: str(path) for path in paths}


def test_expand_uri_paths_single_placeholder(tmp_path):
    f1 = create_nc(tmp_path / "forcing_2020.nc")
    f2 = create_nc(tmp_path / "forcing_2021.nc")
    uri = str(tmp_path / "forcing_{year}.nc")
    results = {
        name: path for path, name in expand_uri_paths(uri, placeholders=["year"])
    }
    assert results == {"2020": str(f1), "2021": str(f2)}


def test_expand_uri_paths_multiple_placeholders(tmp_path):
    f1 = create_nc(tmp_path / "forcing_2020_temp.nc")
    f2 = create_nc(tmp_path / "forcing_2021_precip.nc")
    uri = str(tmp_path / "forcing_{year}_{variable}.nc")
    results = {
        name: path
        for path, name in expand_uri_paths(uri, placeholders=["year", "variable"])
    }
    assert results == {"2020.temp": str(f1), "2021.precip": str(f2)}


def test_expand_uri_paths_wildcard_plus_placeholders(tmp_path):
    f1 = create_nc(tmp_path / "forcing_one_2020_temp.nc")
    f2 = create_nc(tmp_path / "forcing_two_2021_precip.nc")
    uri = str(tmp_path / "forcing_*_{year}_{variable}.nc")
    results = {
        name: path
        for path, name in expand_uri_paths(uri, placeholders=["year", "variable"])
    }
    assert results == {"2020.temp": str(f1), "2021.precip": str(f2)}


def test_expand_uri_paths_placeholders_in_dir(tmp_path):
    dir1 = tmp_path / "2020" / "temp"
    dir1.mkdir(parents=True)
    f1 = create_nc(dir1 / "forcing.nc")
    dir2 = tmp_path / "2021" / "precip"
    dir2.mkdir(parents=True)
    f2 = create_nc(dir2 / "forcing.nc")
    uri = str(tmp_path / "{year}/{variable}/forcing.nc")
    results = {
        name: path
        for path, name in expand_uri_paths(uri, placeholders=["year", "variable"])
    }
    assert results == {"2020.temp": str(f1), "2021.precip": str(f2)}


def test_expand_uri_paths_wildcard_and_placeholder_in_dir(tmp_path):
    (tmp_path / "runA/2020/temp").mkdir(parents=True)
    f1 = create_nc(tmp_path / "runA/2020/temp/forcing.nc")
    (tmp_path / "runB/2021/precip").mkdir(parents=True)
    f2 = create_nc(tmp_path / "runB/2021/precip/forcing.nc")
    uri = str(tmp_path / "*/{year}/{variable}/forcing.nc")
    results = {
        name: path
        for path, name in expand_uri_paths(uri, placeholders=["year", "variable"])
    }
    assert results == {"2020.temp": str(f1), "2021.precip": str(f2)}


def test_expand_uri_paths_fallback_stem(tmp_path):
    f1 = create_nc(tmp_path / "forcing_extra.nc")
    uri = str(tmp_path / "*.nc")  # no placeholder
    results = list(expand_uri_paths(uri))
    assert results[0][1] == f1.stem  # name is filename stem
