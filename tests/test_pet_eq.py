import sys
import numpy as np
from hydromt import DataCatalog
from hydromt.workflows.forcing import pm_fao56, pet, wind


def eq_pet():
    dc = DataCatalog(r"C:\Users\dalmijn\Downloads\tmp_data_export\data_catalog.yml")
    w = dc.get_rasterdataset("era5_daily_zarr")
    dem = dc.get_rasterdataset("era5_orography").squeeze("time").drop("time")
    w["d2m"] -= 273.15
    w2m = wind(dem, wind_u=w["u10"], wind_v=w["v10"], altitude_correction=True)
    p = pm_fao56(
        w["temp"],
        w["temp_max"],
        w["temp_min"],
        w["press_msl"] / 10,
        w["kin"],
        w2m,
        w["d2m"],
        dem,
        "temp_dew",
    )

    # w = w.rename({"d2m":"temp_dew","u10":"wind_u","v10":"wind_v"})

    p2 = pet(w, w["temp"], dem, method="penman-monteith_tdew")

    # pet_old = pet_old.squeeze("x").drop("x")
    return p, p2


if __name__ == "__main__":
    d, d2 = eq_pet()
    sys.stdout.write(f"{d.values[0,0,0]}\n")
    sys.stdout.write(f"{d2.values[0,0,0]}\n")
    a = np.abs(np.subtract(d.values, d2.values))
    print(np.max(a))
