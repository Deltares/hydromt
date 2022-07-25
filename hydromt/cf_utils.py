import os, logging, zipfile
from os.path import isfile, basename, abspath, join, dirname, isdir
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Callable
import re
import shutil
import requests
import glob
import logging
import pandas as pd
import win32com.client
from hydromt.config import configread, configwrite
from configparser import RawConfigParser
from datetime import datetime
import numpy as np
from .cf_xml_utils_extended import CfXmlUtilsExtended

logger = logging.getLogger(__name__)


class FewsUtils(object):
    # root URL of the template CF Config file of Delft-FEWS
    _url = (
        r"https://repos.deltares.nl/repos/BE_Case_Studies/CF_configbase/initialstart/"
    )

    def __init__(self, fews_root, template_path=None, logger=logger):
        """Object containing utils functions and properties to a FEWS application.

        Can be inked to an existing application or start from a local template, or the templates from
        the Delft-FEWS CF application stored at url "https://repos.deltares.nl/repos/BE_Case_Studies/CF_configbase/initialstart/"

        Arguments
        ---------
        fews_root: str, Path
            Path to the FEWS application root. If it does not exist will use local template in template_path or download
            from url.
        template_path: str, Path, optional
            Path to FEWS template files. If not provided, will download an inital start
            template of the FEWS CF application.
        """
        self.logger = logger
        if not isdir(fews_root):
            self.logger.info(
                f"FEWS config not found at {fews_root}, downloading from template"
            )
            self.from_initialstart(fews_root, template_path=template_path)

        # Initialise different subfolders path
        self.fews_root = Path(fews_root)
        self.state_path = self.get_dir(fews_root, ["Config", "ColdStateFiles"])
        self.display_path = self.get_dir(fews_root, ["Config", "DisplayConfigFiles"])
        self.import_path = self.get_dir(fews_root, ["Import"])
        self.map_path = self.get_dir(fews_root, ["Config", "MapLayerFiles"])
        self.module_path = self.get_dir(fews_root, ["Config", "ModuleDataSetFiles"])
        self.region_path = self.get_dir(fews_root, ["Config", "RegionConfigFiles"])
        self.system_path = self.get_dir(fews_root, ["Config", "SystemConfigFiles"])

        # List of templated config files and path
        self.template_configfiles = {
            "Filters": self.region_path,
            "Grids": self.region_path,
            "LocationSets": self.region_path,
            "ModuleInstanceDescriptors": self.region_path,
            "SpatialPlot": self.display_path,
            "TopologyGroup": self.region_path,
            "WhatIfTemplates": self.region_path,
            "WorkflowDescriptors": self.region_path,
        }
        self.locationfiles = {"grids": join(self.map_path)}

        # Initialise dictionnaries with the different models in the FEWS application
        self._models = {}

    @property
    def models(self):
        """Returns dictionary of Models in FEWS config."""
        if len(self._models) == 0:
            self._models = {}
            # self.from_moduledata()  # try reading existing models in config todo implement
        return self._models

    @property
    def keys(self):
        """Returns list of data source names."""
        return list(self.models.keys())

    def __getitem__(self, key):
        return self.models[key]

    def __setitem__(self, key, value):
        if key in self._models:
            self.logger.warning(f"Overwriting data source {key}.")
        return self._models.__setitem__(key, value)

    def __iter__(self):
        return self.models.__iter__()

    def __len__(self):
        return self.models.__len__()

    def update(self, **kwargs):
        """Add model sources."""
        for k, v in kwargs.items():
            self[k] = v

    def add_modeldata(
        self,
        name,
        scheme_version,
        crs,
        **kwargs,
    ):
        """Add model source based on dictionary.

        Parameters
        ----------
        name: dict
            Dictionary of model_sources.
        """
        model_name, region_name, model_version = name.split(".")
        model_dict = {
            "model": model_name,
            "region": region_name,
            "mversion": model_version,
            "sversion": scheme_version,
            #'model_folder': join(f'scheme.{scheme_version}', name),
            "crs": crs,
        }
        for k, v in kwargs.items():
            if k == "shape":
                rows, columns = v
                model_dict["rows"] = rows
                model_dict["columns"] = columns
            elif k == "bounds":
                xmin, ymin, xmax, ymax = v
                model_dict["xmin"] = xmin
                model_dict["ymin"] = ymin
                model_dict["xmax"] = xmax
                model_dict["ymax"] = ymax
            else:
                model_dict[k] = v

        model_dict = {name: model_dict}
        self.update(**model_dict)

    def from_initialstart(self, fews_root, template_path=None):
        """Download an inital start template of the FEWS CF application at url
        "https://repos.deltares.nl/repos/BE_Case_Studies/CF_configbase/initialstart/"
        and save it to fews_root.

        Parameters
        ----------
        fews_root: str or Path
            Path to the new FEWS configuration folder.
        """
        # prepare url and paths
        url = rf"{self._url}"
        folder = fews_root
        if not isdir(folder):
            os.makedirs(folder)
        # Check if template_path is provided and if yes copy to fews_root
        if template_path and isdir(template_path):
            self.copy_basefiles(
                source_dir=Path(template_path),
                destination_dir=Path(fews_root),
                dirs_exist_ok=True,
            )
        # download data
        else:
            with requests.get(url, stream=True) as r:
                if r.status_code != 200:
                    self.logger.error(f"CF templates not found at {url}")
                    return
                self.logger.info(f"Downloading file to {folder}")
                with open(folder, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

    def get_dir(self, base: str, subfolders: List[str] = []) -> Path:
        """
        Static methods returns WindowsPath-object of directory as listed in configuration, if needed supplemented with sub-folders.
        Checks for folder existence and creates new if needed.

        Parameters
        ----------
        base: str
            base folder from which sub-folders are checked/created
        subfolders: list, Optional
            Fews-config subfolders
        Returns
        -------
        folder: Path
            WindowsPath-object
        """
        if not Path(base).exists():
            Path(base).mkdir()
        if len(subfolders) == 0:
            return Path(base)
        else:
            folder = Path(base)
            for sub in subfolders:
                folder = Path(folder).joinpath(sub)
                if not folder.exists():
                    folder.mkdir()
            return folder

    def copy_basefiles(self, source_dir, destination_dir, dirs_exist_ok=True):
        """
        Copies base configuration to a specific goal directory
        :param cf_baseconfig_path: path with base configuration
        :param destination_path: path where final configuration is stored
        :return: copied base files
        """
        assert (
            source_dir.exists()
        ), f"Cannot find source directory for base configuration: {source_dir}"
        assert (
            destination_dir.exists()
        ), f"Cannot find destination directory for base configuration: {destination_dir}"
        shutil.copytree(
            Path(source_dir).resolve(),
            Path(destination_dir).resolve(),
            dirs_exist_ok=dirs_exist_ok,
        )

    def add_template_configfiles(
        self, model_source, model_templates=None, variables=[]
    ):
        """
        Update and add template config files in model_templates folder to the fews root
        for model_source instance.

        Parameters
        ----------
        model_source: str
            Model source in FewsUtils model catalog.
        model_templates: str, Path, optional
            Folde containing template config files for tag replacement. If not provided,
            download from url.
        variables: list, optional
            List of additional variables to add to SpatialDisplay.xml. Template should be available.
        """
        model = self.models[model_source]
        # Check if model_templates exist else download
        if not isdir(model_templates):
            # TODO download
            return
        scheme = model.get("sversion")
        mod = model.get("model")
        region = model.get("region")
        mversion = model.get("mversion")

        # First loop over variables to fill in the SpatialDisplay template before tag replacement
        if len(variables) > 0:
            # Copy and create a temporary template for model Saptial Display
            sdis_fn = join(
                model_templates,
                f"SpatialPlot_{mod}_parameters",
                "SpatialPlot_wflow.{region}.{mversion}_base.xml",
            )
            sdis_fn_temp = join(
                model_templates, "SpatialPlot_wflow.{region}.{mversion}.xml"
            )
            shutil.copy(sdis_fn, sdis_fn_temp)
            # Variable templates folder
            var_fns = glob.glob(
                join(
                    model_templates,
                    f"SpatialPlot_{mod}_parameters",
                    "SpatialPlot_*.xml",
                )
            )
            for var in variables:
                # Read template file
                var_fn = [f for f in var_fns if f.endswith(f"{var}.xml")]
                if len(var_fn) > 0:
                    varf = open(var_fn[0], "r")
                    sdis = open(sdis_fn_temp, "a")
                    sdis.write(varf.read())
                    varf.close()
                    sdis.close()
            # Add two final lines
            sdis = open(sdis_fn_temp, "a")
            sdis.write("	</gridPlotGroup>\n")
            sdis.write("</gridPlotGroups>")
            sdis.close()

        # Copy and tag replacement for each *.xml file in model_templates folder
        files = glob.glob(join(model_templates, "*.xml"))
        for file in files:
            filename = basename(file)
            filetype = filename.split("_")[0]
            dest_folder = self.template_configfiles[filetype]
            # dest_filename = eval(f"f'{basename(file)}'")
            dest_file = join(
                dest_folder,
                f"scheme.{scheme}",
                filename.format(region=region, mversion=mversion),
            )
            self.replace_tags_by_file(
                source_file=Path(file),
                destination_file=Path(dest_file),
                tag_dict=model,
            )
        # Remove temporary spatial display template
        if len(variables) > 0:
            os.remove(sdis_fn_temp)

    def add_locationsfiles(self, model_source, model_templates):
        """
        Update location files (csv) in the fews root
        for model_source instance.

        Parameters
        ----------
        model_source: str
            Model source in FewsUtils model catalog.
        model_templates: str, Path, optional
            Folde containing template config files for tag replacement. If not provided,
            download from url.
        """
        model = self.models[model_source]
        mod = model.get("model")
        region = model.get("region")
        mversion = model.get("mversion")
        locid = f"{mod}.{region}.{mversion}"
        for fname in self.locationfiles:
            fpath = self.locationfiles[fname]
            fname = join(fpath, mod, f"{mod}_{fname}.csv")
            if isfile(fname):
                df = pd.read_csv(fname, sep=";")
                if locid not in df["ID"]:
                    df = df.append({"ID": locid}, ignore_index=True)
                df.drop_duplicates().to_csv(fname, sep=";", index=False)

    def add_spatialplots(self, model_source):
        """
        Update SpatialDisplay.xml with spatial plots in the fews root
        for model_source instance.

        Parameters
        ----------
        model_source: str
            Model source in FewsUtils model catalog.
        """
        t = CfXmlUtilsExtended(logger=logger)

        model = self.models[model_source]
        mod = model.get("model")
        region = model.get("region")
        scheme = model.get("sversion")
        mversion = model.get("mversion")
        fpath = self.template_configfiles["SpatialPlot"]

        # model instance
        fname = join(
            fpath,
            f"scheme.{scheme}",
            f"SpatialPlot_{model_source}.xml",
        )

        # get the gridplotgroup ids
        gridplotgroups = t.get_ids_from_xml(fname, "gridPlotGroup")

        #  SpatialDisplay.xml
        fname = join(
            fpath,
            f"SpatialDisplay.xml",
        )

        # insert spatial extent
        t.insert_extra_extent_into_spatial_display_xml(
            spatial_display_file=fname,
            region=region,
            top=model.get("ymax"),
            bottom=model.get("ymin"),
            left=model.get("xmin"),
            right=model.get("xmax"),
        )

        # inser gridplotgroup ids
        for groupid in gridplotgroups:
            t.append_gridplotgroup_to_spatial_display_xml(
                spatial_display_file=fname, groupId=groupid
            )

    def add_topologygroups(self, model_source):
        """
        Update Topology.xml with spatial plots in the fews root
        for model_source instance.

        Parameters
        ----------
        model_source: str
            Model source in FewsUtils model catalog.
        """
        t = CfXmlUtilsExtended(logger=logger)

        model = self.models[model_source]
        mod = model.get("model")
        region = model.get("region")
        scheme = model.get("sversion")
        mversion = model.get("mversion")
        fpath = self.template_configfiles["TopologyGroup"]

        # model instance for forcing
        fname = join(
            fpath,
            f"scheme.{scheme}",
            f"TopologyGroup_{model_source}_forc.xml",
        )

        # get the gridplotgroup ids
        groups = t.get_ids_from_xml(fname, "group")

        # model instance
        fname = join(
            fpath,
            f"scheme.{scheme}",
            f"TopologyGroup_{model_source}.xml",
        )

        # get the gridplotgroup ids
        groups.extend(t.get_ids_from_xml(fname, "group"))

        #  SpatialDisplay.xml
        fname = join(
            fpath,
            f"Topology.xml",
        )

        # insert groupid
        for groupid in groups:
            t.append_group_to_topology_xml(topology_file=fname, groupId=groupid)

    def update_explorer(self, model_source):
        """
        Update Exploer.xml with extent for model_source instance.

        Parameters
        ----------
        model_source: str
            Model source in FewsUtils model catalog.
        """
        t = CfXmlUtilsExtended(logger=logger)

        model = self.models[model_source]
        mod = model.get("model")
        region = model.get("region")
        scheme = model.get("sversion")
        mversion = model.get("mversion")
        fpath = self.system_path

        # model instance
        fname = join(
            fpath,
            f"Explorer.xml",
        )

        # insert spatial extent
        t.insert_extra_extent_into_explorer_xml(
            explorer_file=fname,
            region=region,
            top=model.get("ymax"),
            bottom=model.get("ymin"),
            left=model.get("xmin"),
            right=model.get("xmax"),
        )

    def update_globalproperties(self, model_source, model_templates):
        """
        Update global.properties with T0 and extent for model_source instance.

        Parameters
        ----------
        model_source: str
            Model source in FewsUtils model catalog.
         model_templates: str, Path, optional
             Folde containing template global properties fper model
        """

        model = self.models[model_source]
        mod = model.get("model")
        region = model.get("region")
        scheme = model.get("sversion")
        mversion = model.get("mversion")
        T0 = model.get("T0")
        fpath = self.fews_root

        # read global properties
        fname = join(
            fpath,
            f"global.properties",
        )
        globalproperty = configread(
            fname,
            cf=RawConfigParser,
            noheader=True,
            abs_path=False,
        )

        # update T0
        T0_format = "%d-%m-%Y"
        globalproperty_T0 = datetime.strptime(
            globalproperty.get("T0", datetime.strftime(datetime.today(), T0_format)),
            T0_format,
        )
        if T0 is None:
            T0 = datetime.today()
        globalproperty_T0 = min(globalproperty_T0, T0)
        globalproperty["T0"] = datetime.strftime(globalproperty_T0, T0_format)

        # update bbox
        xmin = globalproperty.get("BBOX_LEFT", np.inf)
        ymin = globalproperty.get("BBOX_BOTTOM", np.inf)
        xmax = globalproperty.get("BBOX_RIGHT", -np.inf)
        ymax = globalproperty.get("BBOX_TOP", -np.inf)

        globalproperty["BBOX_LEFT"] = min(xmin, model.get("xmin"))
        globalproperty["BBOX_BOTTOM"] = min(ymin, model.get("ymin"))
        globalproperty["BBOX_RIGHT"] = max(xmax, model.get("xmax"))
        globalproperty["BBOX_TOP"] = max(ymax, model.get("ymax"))

        # update others from template
        fname_model = join(
            model_templates,
            f"wflow_global.properties",
        )
        globalproperty_model = configread(
            fname_model,
            cf=RawConfigParser,
            noheader=True,
            abs_path=False,
        )
        for k, v in globalproperty_model.items():
            globalproperty[k] = v

        # write global properties
        configwrite(fname, globalproperty, cf=RawConfigParser, noheader=True)

    def replace_tags_by_file(self, source_file, destination_file, tag_dict):
        """
        replaces tags in a source file templates and copies the filled source file to a destination file
        :param source_file: path to file that contains tags and therefore serves as template
        :param destination_file: path to file with tags from source file replaced by values
        :return: specific filled templates
        """
        assert source_file.exists(), f"Cannot find source file: {source_file}"
        if not isdir(os.path.dirname(destination_file)):
            os.makedirs(os.path.dirname(destination_file))

        destination_file_add = shutil.copy(
            source_file, destination_file, follow_symlinks=True
        )
        file_name = basename(destination_file_add)

        with open(destination_file_add, "r") as f:
            template = f.read()
            template_filled = template
            file_name_filled = file_name
            for key, value in tag_dict.items():
                value = str(value)
                file_name_filled = re.sub("{" + key + "}", value, file_name_filled)
                template_filled = re.sub("{" + key + "}", value, template_filled)
            with open(destination_file_add, "w") as ff:
                ff.write(template_filled)

    def create_cf_link(self, fews_binaries=None):
        """ "
        Creates a FEWS (CF) link to fews_root

        Parameters
        ----------
        binaries_path: str, Path, optional
            Path to FEWS binaries. If not provided assume FEWS bin folder in fews root.
        """

        if fews_binaries == None:
            fews_binaries = join(self.fews_root, "bin")
            self.logger.warning(
                f"Use relative path to FEWS binaries {fews_binaries} to create FEWS .lnk application"
            )

        shell = win32com.client.Dispatch("WScript.Shell")

        link_name = self.fews_root.name
        link = str(self.fews_root.joinpath(str(link_name) + ".lnk"))
        shortcut = shell.CreateShortCut(link)
        shortcut.Targetpath = str(fews_binaries.joinpath("./windows/Delft-FEWS.exe"))

        shortcut.Arguments = f"-Dregion.home=."
        shortcut.save()

    #    return link

    # def write_ini_file(sections: Dict, ini_file: Path):
    #     """
    #     Reads parameter file and returns dataframe with:
    #     ID, NAME, SHORT_NAME, GROUP, TYPE, UNIT, VALUE_RESOLUTION
    #     :param ini:
    #     :return:
    #     """
    #     with open(ini_file, "w") as outf:
    #         for section in sections:
    #             outf.write("[" + section + "]\n")
    #             for key in sections[section].keys():
    #                 outf.write(key + "=" + sections[section][key] + "\n")

    # def add_wgs84_projection_file(fname: Path):
    #     """
    #     Add missing projection file, expressed in WGS 1984
    #     :param fname:
    #     :return:
    #     """
    #     proj_str = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
    #     proj_file = open(fname, "w")
    #     proj_file.write(proj_str)
    #     proj_file.close()

    # def replace_tags_by_filetree(
    #     source_dir_templ, destination_dir, tag_dict, extensions
    # ):
    #     """
    #     replaces tags in all source file templates in the file tree and save the filled templates in the respective destination pth
    #     :param source_file: source file that contains tags and therefore serves as template
    #     :param destination_file: file with tags from source file replaced by values
    #     :return: all filled templates
    #     """
    #     assert (
    #         source_dir_templ.exists()
    #     ), f"Cannot find source directory for templates: {source_dir_templ}"
    #     assert (
    #         destination_dir.exists()
    #     ), f"Cannot find destination directory: {destination_dir}"

    #     tag_files = []
    #     for path in source_dir_templ.rglob("*"):
    #         if path.suffix in extensions:
    #             tag_files.append(path)

    #     for file_path in tag_files:
    #         tag_files_paths_filled = file_path
    #         for key, value in tag_dict.items():
    #             if key in str(tag_files_paths_filled):
    #                 tag_files_paths_filled = re.sub(
    #                     "{" + key + "}", value, str(tag_files_paths_filled)
    #                 )
    #                 if not "{" in tag_files_paths_filled:
    #                     source_file = file_path
    #                     relative_path = os.path.relpath(source_file, source_dir_templ)
    #                     destination_file = Path(destination_dir / relative_path)
    #                     replace_tags_by_file(
    #                         source_file, destination_file, tag_dict, extensions
    #                     )
    #             else:  # if no tag that needs to be replaced present in file_path
    #                 if key == list(tag_dict)[-1]:
    #                     source_file = file_path
    #                     relative_path = os.path.relpath(source_file, source_dir_templ)
    #                     destination_file = Path(destination_dir / relative_path)
    #                     os.makedirs(os.path.dirname(destination_file), exist_ok=True)
    #                     shutil.copy(source_file, destination_file, follow_symlinks=True)
