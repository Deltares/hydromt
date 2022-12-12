import sys
import numpy as np
from hydromt import DataCatalog
from hydromt.workflows.forcing import pm_fao56, pet_penman_monteith, wind

def eq_pet():
	dc = DataCatalog(r"C:\Users\dalmijn\Downloads\tmp_data_export\data_catalog.yml")
	w = dc.get_rasterdataset("era5_daily_zarr")
	dem = dc.get_rasterdataset("era5_orography").squeeze("time").drop("time")
	dew = w["d2m"] - 273.15
	w2m = wind(
		dem,
		wind_u=w["u10"],
		wind_v=w["v10"],
		altitude_correction=True
		)
	pet = pm_fao56(
		w["temp"],
		w["temp_max"],
		w["temp_min"],
		w["press_msl"],
		w["kin"],
		w2m,
		dew,
		dem,
		"temp_dew"
		)

	pet_old = pet_penman_monteith(
		w["temp"],
		w["temp_max"],
		w["temp_min"],
		w["press_msl"],
		w["kin"],
		w2m,
		dew,
		dem,
		"temp_dew"
		)
	# pet_old = pet_old.squeeze("x").drop("x")
	return pet, pet_old

if __name__ == "__main__":
	d,d2 = eq_pet()
	sys.stdout.write(f"{d.values[0,0,0]}\n")
	print(d2.values[0,0,0])
	print(d.values)
	a = np.abs(np.subtract(d.values,d2.values))
	print(np.mean(a))