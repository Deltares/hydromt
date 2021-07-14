"""
This module contains different functions and classes to connect to the BlueEarth Computational Framework.
"""

from pathlib import Path
import shutil
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict
import pandas as pd

from . import xml_utils, io

logger = logging.getLogger(__name__)


def add_wgs84_projection_file(fname: Path):
    """
    Add missing projection file, expressed in WGS 1984
    """
    proj_str = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
    proj_file = open(fname, "w")
    proj_file.write(proj_str)
    proj_file.close()


class DescriptorsXml:
    """
    Provides basic functions to Create ModuleInsatnceDescriptors, Workflowdescriptors and TopologyGroups
    """

    def __init__(self, config: Dict, logger: logging.Logger = logger):
        """
        initialisation

        Parameters
        ----------
        config: dict
            configuration Dictionary with keys:

                * networkid: identifier of the network/model
                * codebase: name of the modelcode base (ribasim, wflow etc)
                * timestep: timestep of the model
                * cf_application_folder: application (Region_Home) of the Fews-application, holding the Config, Modules and a Preparation folder
                * cf_config_folder: application/Config folder of the Fews-application
        """
        self.logger = logger
        self.ini = config
        self.codebase = self.ini["codebase"]
        self.networkid = self.ini["networkid"]
        self.timestep = self.ini["timestep"]
        self.region_template_dir = io.get_dir(
            base=self.ini["cf_application_folder"],
            subfolders=[
                "Modules",
                self.codebase,
                "system",
                "cf_configurator",
                "config_templates",
                "RegionConfigFiles",
            ],
        )
        self.region_dir = io.get_dir(
            base=self.ini["cf_config_folder"], subfolders=["RegionConfigFiles"]
        )
        self.region_ntw_dir = io.get_dir(
            base=self.ini["cf_config_folder"],
            subfolders=["RegionConfigFiles", self.networkid],
        )

        ET.register_namespace("", "http://www.wldelft.nl/fews")

    def create_module_instance_sets_xml(self):
        """Copies a ModuleInstanceSets.xml from a template folder to the Config/RegionConfigFiles folder"""

        dest_file = self.region_dir.joinpath("ModuleInstanceSets.xml")
        if not dest_file.exists():
            pickup_file = self.region_template_dir.joinpath(("ModuleInstanceSets.xml"))
            try:
                shutil.copy2(pickup_file, dest_file)
            except Exception as e:
                msg = "Cannot find template file: {}, Exception {}".format(
                    pickup_file, e
                )
                self.logger.error(msg)
                raise msg

    def create_module_instance_descriptors_xml(self, descriptors: List):
        """
        Create ModuleInstanceDescriptor_<codebase>_<networkid>.xml for the list of descriptors.

        Parameters
        ----------
        descriptors: list
            List of descriptors to add
        """
        self.create_module_instance_sets_xml()
        xml_tree = xml_utils.create_fews_xml_root(
            schema="moduleInstanceDescriptors", version="1.0"
        )
        xml_tree.append(
            ET.Comment("CF descriptors for network {}".format(self.networkid))
        )
        for descriptor in descriptors:
            ET.SubElement(xml_tree, "moduleInstanceDescriptor", {"id": descriptor})
        descr_file = self.region_ntw_dir.joinpath(
            "ModuleInstanceDescriptors_{}__{}.xml".format(self.codebase, self.networkid)
        )

        xml_utils.serialize_xml(xml_tree, descr_file)
        return xml_tree

    def append_module_instance_descriptors(self, descr_to_append: List):
        """
        Create ModuleInstanceDescriptor_<codebase>_<networkid>.xml for the list of descriptors.

        Parameters
        ----------
        descr_to_append: list
            List of descriptors to add
        """

        fname = self.region_ntw_dir.joinpath(
            "ModuleInstanceDescriptors_{}__{}.xml".format(self.codebase, self.networkid)
        )
        try:
            tree = ET.parse(fname)
            xml_root = tree.getroot()
        except:
            msg = "Cannot parse {}. Please create moduleinstancedescriptors before calling this function".format(
                fname
            )
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        available = [
            instance.attrib["id"]
            for instance in xml_root.findall(
                "{http://www.wldelft.nl/fews}moduleInstanceDescriptor"
            )
        ]
        for descr in descr_to_append:
            if descr not in available:
                ET.SubElement(xml_root, "moduleInstanceDescriptor", {"id": descr})
        xml_utils.serialize_xml(xml_root, fname)

    def _add_workflow_descriptor(
        self, parent_tree: ET.ElementTree, task: str, descr: str, properties: List = []
    ):
        """
        Adds workflow descriptor with identifier task_codebase_networkid
        Includes properties if specified

        Parameters
        ----------
        parent_tree: ET.ElementTree
            xml-Object to append
        task: str
            workflow task name
        descr: str
            workflow description
        properties: list
            optional properties to include
        """
        wfd_tree = ET.SubElement(
            parent_tree,
            "workflowDescriptor",
            {
                "id": "{}{}__{}".format(task, self.codebase.title(), self.networkid),
                "name": "{} {} model {}".format(
                    task, self.codebase.title(), self.networkid
                ),
            },
        )
        xml_utils.add_child_with_text(wfd_tree, "description", descr)
        xml_utils.add_child_with_text(
            wfd_tree, "workflowFileName", "{}{}".format(task, self.codebase.title())
        )
        ET.SubElement(wfd_tree, "runExpiryTime", {"unit": "day", "multiplier": "365"})
        if len(properties) > 0:
            prop_tree = ET.SubElement(wfd_tree, "properties")
            for prop in properties:
                ET.SubElement(prop_tree, "string", prop)
        return parent_tree

    def create_workflow_descriptors_xml(self, tasks: Dict):
        """
        Create WorkflowDescriptors with tasks and save to file (with _code_networkid)

        Parameters
        ----------
        tasks: dict
            Dict of workflow descriptor tasks with 'desc' (description as str) and 'prop'(list of workflow property dicts)
        """
        xml_tree = xml_utils.create_fews_xml_root(
            schema="workflowDescriptors", version="1.0"
        )
        xml_tree.append(
            ET.Comment(
                "{} CF descriptors for network {}".format(
                    self.codebase.title(), self.networkid
                )
            )
        )
        # add validate descriptor with moduledataset properties
        for task in tasks:
            self._add_workflow_descriptor(
                parent_tree=xml_tree,
                task=task.title(),
                descr=tasks[task]["desc"],
                properties=tasks[task]["prop"],
            )
        xml_utils.serialize_xml(
            xml_tree,
            self.region_ntw_dir.joinpath(
                "WorkflowDescriptors_{}__{}.xml".format(self.codebase, self.networkid)
            ),
        )

    def append_properties_to_workflowdescriptors_xml(
        self, wftask: str = "", properties_to_append: List[Dict] = []
    ):
        """
        Appends WorkflowDescriptors task with properties and saves to file (with _code_networkid)

        Parameters
        ----------
        wftask: str, Optional
            workflow identifier string
        properties_to_append: list, Optional
            list of property dicts to add
        """
        fname = self.region_ntw_dir.joinpath(
            "WorkflowDescriptors_{}__{}.xml".format(self.codebase, self.networkid)
        )
        try:
            tree = ET.parse(fname)
        except:
            msg = "Cannot parse {}. Please create workflowdescriptors before calling this function".format(
                fname
            )
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        root = tree.getroot()
        for workflowDescriptor in root.iter(
            "{http://www.wldelft.nl/fews}workflowDescriptor"
        ):
            if wftask.lower() in workflowDescriptor.attrib["id"].lower():
                for elem in workflowDescriptor.iter():
                    if elem.tag == "{http://www.wldelft.nl/fews}properties":
                        existing_prop_keys = [prop.attrib["key"] for prop in elem]
                        properties_to_append = [
                            x
                            for x in properties_to_append
                            if x["key"] not in existing_prop_keys
                        ]
                        for x in properties_to_append:
                            ET.SubElement(elem, "string", x)

        xml_utils.serialize_xml(root, fname)

    def _append_topologygroup_to_topology_xml(self):
        """
        Reads Topology.xml from RegionConfigFiles folder and adds topologygroup of networkid
        """
        if self.region_dir.joinpath("Topology.xml").exists():
            try:
                tree = ET.parse(self.region_dir.joinpath("Topology.xml"))
                root = tree.getroot()
            except Exception as e:
                msg = "Could not parse Topology.xml file in folder: {} raised Exception {}".format(
                    self.region_dir, e
                )
                self.logger.error(msg)
                raise msg
        else:
            try:
                tree = ET.parse(self.region_template_dir.joinpath("Topology.xml"))
                root = tree.getroot()
            except Exception as e:
                msg = "Could not find template Topology.xml file in folder: {} raised Exception {}".format(
                    self.region_template_dir, e
                )
                self.logger.error(msg)
                raise msg

        existing_networks = [
            nodes.attrib["id"]
            for nodes in root.findall("{http://www.wldelft.nl/fews}nodes")
        ]
        if "all_{}".format(self.networkid) not in existing_networks:
            nodes_tree = ET.SubElement(
                root,
                "nodes",
                {"id": "all_{}".format(self.networkid), "name": self.networkid},
            )
            xml_utils.add_child_with_text(nodes_tree, "groupId", self.networkid)
            ET.SubElement(nodes_tree, "graceTime", {"unit": "hour", "multiplier": "1"})
        xml_utils.serialize_xml(root, self.region_dir.joinpath("Topology.xml"))

    def create_topologygroup_xml(self, nodes: Dict):
        """
        Creates TopologyGroup.xml file with nodes (codebase_networkid) listing the contents of nodes dictionary

        Parameters
        ----------
        nodes: dict
            dict holding string entries for keys id, name, workflowId, whatIfTemplateId (optional)
        """

        xml_tree = xml_utils.create_fews_xml_root(schema="topologyGroup")
        xml_tree.append(
            ET.Comment(
                "{}-CF topology for network {}".format(self.codebase, self.networkid)
            )
        )
        group_tree = ET.SubElement(xml_tree, "group", {"id": self.networkid})
        nodes_tree = ET.SubElement(
            group_tree,
            "nodes",
            {
                "id": "{}_{}".format(self.codebase, self.networkid),
                "name": self.codebase.title(),
            },
        )
        for node in nodes:
            node_tree = ET.SubElement(
                nodes_tree, "node", {"id": node["id"], "name": node["name"]}
            )
            xml_utils.add_child_with_text(node_tree, "workflowId", node["workflowId"])
            if "whatIfTemplateId" in node:
                xml_utils.add_child_with_text(
                    node_tree, "whatIfTemplateId", node["whatIfTemplateId"]
                )
            xml_utils.add_child_with_text(node_tree, "localRun", "true")
            xml_utils.add_child_with_text(node_tree, "saveLocalRunEnabled", "true")

        xml_utils.serialize_xml(
            xml_tree,
            self.region_ntw_dir.joinpath("TopologyGroup_{}.xml".format(self.networkid)),
        )

        self._append_topologygroup_to_topology_xml()


class ExplorerXml:
    """
    Updates Explorer after first model imported
    """

    def __init__(self, config: Dict, logger: logging.Logger):
        """
        initialisation

        Parameters
        ----------
        config: dict
            configuration Dictionary with keys:

                * networkid: identifier of the network/model
                * codebase: name of the modelcode base (ribasim, wflow etc)
                * (optional) top, bottom, left, right: bounding box extend in WGS1984
                * cf_config_folder: Region_Home/Config folder of the Fews-application
        """
        self.logger = logger

        self.ini = config
        # model and network name
        self.networkid = self.ini["networkid"]
        self.codebase = self.ini["codebase"]
        # extents
        self.includeBoundingBox = True if "top" in self.ini else False

        self.system_cfg_dir = io.get_dir(
            base=self.ini["cf_config_folder"], subfolders=["SystemConfigFiles"]
        )
        ET.register_namespace("", "http://www.wldelft.nl/fews")

    def __task_spatialdisplay__(self):
        task_str = """
                    <explorerTask name="Spatial Results">
                        <mnemonic>p</mnemonic>
                        <displayConfigFileName>SpatialDisplay</displayConfigFileName>
                        <toolbarTask>true</toolbarTask>
                        <menubarTask>true</menubarTask>
                        <allowMultipleInstances>true</allowMultipleInstances>
                        <accelerator>ctrl R</accelerator>
                        <loadAtStartup>true</loadAtStartup>
                    </explorerTask>
                    """
        task_tree = ET.fromstring(task_str)
        return task_tree

    def __task_whatif__(self):
        task_str = """
                    <explorerTask name="Case Management">
                        <iconFile>data_tools.png</iconFile>
                        <taskClass>nl.wldelft.fews.gui.plugin.whatif.editor.WhatIfEditor</taskClass>
                        <toolbarTask>true</toolbarTask>
                        <menubarTask>true</menubarTask>
                        <loadAtStartup>true</loadAtStartup>
                    </explorerTask>
                    """
        task_tree = ET.fromstring(task_str)
        return task_tree

    def __task_modifier__(self):
        task_str = """
                    <explorerTask name="Interventions">
                        <mnemonic>M</mnemonic>
                        <arguments>ModifiersPlugin</arguments>
                        <taskClass>nl.wldelft.fews.gui.plugin.modifiersdisplay.ModifiersPanel</taskClass>
                        <toolbarTask>true</toolbarTask>
                        <menubarTask>true</menubarTask>
                        <accelerator>ctrl M</accelerator>
                        <loadAtStartup>true</loadAtStartup>
                    </explorerTask>
                    """
        task_tree = ET.fromstring(task_str)
        return task_tree

    def _add_extra_extent(self, xml_root: ET.ElementTree):
        """
        Function to add extra extent to geomap (after default extent) if Basin is not yet defined

        Parameters
        ----------
        xml_root: ET.ElementTree
            Spatial Display root
        """

        def _get_index(root, element):
            for idx, child in enumerate(root):
                if element == child:
                    return idx
            else:
                raise ValueError(
                    "No '%s' tag found in '%s' children" % (element.tag, root.tag)
                )

        def _new_extra_extent():
            extra_tree = ET.Element("extraExtent", {"id": self.networkid})
            xml_utils.add_child_with_text(extra_tree, "left", self.ini["left"])
            xml_utils.add_child_with_text(extra_tree, "right", self.ini["right"])
            xml_utils.add_child_with_text(extra_tree, "top", self.ini["top"])
            xml_utils.add_child_with_text(extra_tree, "bottom", self.ini["bottom"])

            return extra_tree

        geomap = xml_root.find("{http://www.wldelft.nl/fews}map")
        if len(geomap) > 0:
            extra_extents = geomap.findall("{http://www.wldelft.nl/fews}extraExtent")
            if extra_extents is not None:
                existing_extents = [
                    extra_extent.attrib["id"] for extra_extent in extra_extents
                ]
                if not self.networkid in existing_extents:
                    elem_index = _get_index(
                        geomap, geomap.find("{http://www.wldelft.nl/fews}defaultExtent")
                    )
                    geomap.insert(elem_index + 1, _new_extra_extent())

        return None

    def _add_child_layer_group(self, xml_root: ET.ElementTree, shape_files: List):
        """
        Function to add network child layer group to a parent layer group

        Parameters
        ----------
        xml_root: ET.ElementTree
            spatial display
        shape_files: list
            List of dicts refencing shapefiles to include. Dict should hold id, file (name), pointSize (optional for nodes), lineColor (optional for links)
        """

        def _find_layergroup(xml_root: ET.ElementTree):
            """
            find layer group in geomap referencing code base to expand
            """
            geomap = xml_root.find("{http://www.wldelft.nl/fews}map")
            if len(geomap) > 0:
                toplayergroups = geomap.findall(
                    "{http://www.wldelft.nl/fews}layerGroup"
                )
                if len(toplayergroups) > 0:
                    for toplayergroup in toplayergroups:
                        if toplayergroup.attrib["id"].lower() == self.codebase:
                            return toplayergroup
                return ET.SubElement(
                    geomap,
                    "layerGroup",
                    {"id": self.codebase, "name": self.codebase.title()},
                )
            self.logger.warning("No GeoMap available, cannot add layergroup")
            return None

        def _add_network_layergroup(layergroups: ET.Element, shape_files: List):
            """
            Add shape files to layergroudps element of this codebase

            Parameters
            ----------
            layergroups:
                layergroups element to expand
            shape_files: list
                List of dicts refencing shapefiles to include. Dict should hold id, file (name), pointSize (optional for nodes), lineColor (optional for links)
            """
            networklayergroup = ET.SubElement(
                layergroups, "layerGroup", {"id": self.networkid}
            )
            for shape in shape_files:
                layer = ET.SubElement(
                    networklayergroup, "esriShapeLayer", {"id": shape["id"]}
                )
                xml_utils.add_child_with_text(layer, "file", shape["file"])
                xml_utils.add_child_with_text(layer, "geoDatum", "WGS 1984")
                xml_utils.add_child_with_text(layer, "visible", "true")
                if "pointSize" in shape:
                    xml_utils.add_child_with_text(
                        layer, "pointSize", shape["pointSize"]
                    )
                elif "lineColor" in shape:
                    xml_utils.add_child_with_text(
                        layer, "lineColor", shape["lineColor"]
                    )

        toplayergroup = _find_layergroup(xml_root)
        if toplayergroup is None:
            _add_network_layergroup(layergroups=toplayergroup, shape_files=shape_files)
        else:
            layergroups = toplayergroup.findall(
                "{http://www.wldelft.nl/fews}layerGroup"
            )
            existing_layergroups = [
                layergroup.attrib["id"] for layergroup in layergroups
            ]
        if not self.networkid in existing_layergroups:
            _add_network_layergroup(layergroups=toplayergroup, shape_files=shape_files)
        # resulting_layergroups = [layergroup.attrib['id'] for layergroup in layergroups]
        return None

    def _append_explorertasks(self, xml_root: ET.ElementTree):
        """
        Append explorertasks that require a locationset of the model to be available in order to prevent config-errors
        If needed adds explorer task references to Spatial Display, WhatIfDisplay and ModifiersDisplay

        Parameters
        ----------
        xml_root: ET.ElementTree
            root element to extend
        """
        hasSpatialDisplay = False
        hasModifierDisplay = False
        hasWhatIfDisplay = False
        explorerTasks = xml_root.find("{http://www.wldelft.nl/fews}explorerTasks")

        existing_tasks = [
            tasks.attrib["name"].lower()
            for tasks in explorerTasks.findall(
                "{http://www.wldelft.nl/fews}explorerTask"
            )
        ]
        for task in existing_tasks:
            if "spatial" in task:
                hasSpatialDisplay = True
            elif "whatif" in task:
                hasWhatIfDisplay = True
            elif "case" in task:
                hasWhatIfDisplay = True
            elif "modifier" in task:
                hasModifierDisplay = True
            elif "intervention" in task:
                hasModifierDisplay = True
            elif "measure" in task:
                hasModifierDisplay = True
        if not hasSpatialDisplay:
            explorerTasks.append(self.__task_spatialdisplay__())
        if not hasWhatIfDisplay:
            explorerTasks.append(self.__task_whatif__())
        if not hasModifierDisplay:
            explorerTasks.append(self.__task_modifier__())
        return None

    def append_explorer_xml(self, shapefiles: List):
        """
        Main function to append network to Explorer file (extent, layergroup and tasks)
        """

        fname = self.system_cfg_dir.joinpath("Explorer.xml")

        tree = ET.parse(fname)
        xml_root = tree.getroot()
        if self.includeBoundingBox:
            self._add_extra_extent(xml_root=xml_root)
        self._add_child_layer_group(xml_root=xml_root, shape_files=shapefiles)
        self._append_explorertasks(xml_root=xml_root)
        # write
        xml_utils.serialize_xml(xml_root, fname)


# TODO: generalize and remove RIBASIM specific stuff
class FiltersXml:
    """
    Provides basic functions to create ribasim filters
    """

    def __init__(self, config: Dict, logger: logging.Logger):
        """
        initialisation

        Parameters
        ----------
        config: dict
            dictionary with keys:

                * networkid: identifier of the network/model
                * codebase: name of the modelcode base (ribasim, wflow etc)
                * timestep: timestep of the model
                * featuretypes_df: data frame of featuretypes with columns ftype, nice, node_group
                * parameters_df: dataframe of (Fews)parameters with columns FEATURE_TYPE, INT_PARID
                * cf_application_folder: application (Region_Home) of the Fews-application, holding the Config, Modules and a Preparation folder
                * cf_config_folder: application/Config folder of the Fews-application

        """
        self.logger = logger

        self.ini = config
        # model and network refernce
        self.networkid = self.ini["networkid"]
        self.timestep = "nonequidistant"  # self.ini['timestep']
        self.codebase = self.ini["codebase"]
        # the feature and featureNames
        # NOTE! dropping nan values for features and featureNames in ntr_df: logger.warning('')?
        self.features = self.ini["featuretypes_df"].ftype
        self.featureNames = self.ini["featuretypes_df"].nice

        self.fgroups_dict = (
            self.ini["featuretypes_df"]
            .pivot_table(columns="ftype_group", values=["ftype"], aggfunc=list)
            .squeeze()
            .to_dict()
        )
        # output parameters
        self.param_dfs = self.ini["parameters_df"]
        # all available timesteps
        self.region_dir = io.get_dir(
            base=self.ini["cf_config_folder"], subfolders=["RegionConfigFiles"]
        )
        self.region_rbs_dir = io.get_dir(
            base=self.ini["cf_config_folder"],
            subfolders=["RegionConfigFiles", self.codebase, "general"],
        )

        self.region_ntw_dir = io.get_dir(
            base=self.ini["cf_config_folder"],
            subfolders=["RegionConfigFiles", self.networkid],
        )
        ET.register_namespace("", "http://www.wldelft.nl/fews")

    def _add_nested_ftype_filter(
        self, root_tree: ET.ElementTree, fgroup: str, children: list
    ):
        """
        Creates new nested filter with parent - children pair

        Parameters
        ----------
        fgroup: str
        make the filter id: networkid__group_timestep, name: fgroup
        children: list
            make the child filter with foreign_key: parent__child
        """
        parent_tree = ET.SubElement(
            root_tree, "filter", {"id": self.networkid + "__" + fgroup, "name": fgroup}
        )
        #        parent_tree = ET.SubElement(root_tree, 'filter',
        #                                    {'id': self.networkid + '__'+ fgroup + '_' + self.timestep, 'name': fgroup})
        for child in children:
            ET.SubElement(parent_tree, "child", {"foreignKey": child})
        return None

    def _add_nested_fgroup_filter(
        self, root_tree: ET.ElementTree, parent: str, fgroups: dict
    ):
        """
        Creates nested filter of featuretype groups (children) for network parent .

        Within each group it calls the parent-child nesting function to create
        nested filters (children) for featuretypes in the parent group

        Parameters
        ----------
        parent: str
            make the filter id: parent__all, name: model: parent
        children: list
            make the child filter with foreign_key: parent__child
        """
        parent_tree = ET.SubElement(
            root_tree, "filter", {"id": parent + "__all", "name": parent}
        )
        for fgroup in fgroups:
            ET.SubElement(parent_tree, "child", {"foreignKey": parent + "__" + fgroup})
        #            ET.SubElement(parent_tree, 'child', {'foreignKey': parent + '__' + fgroup + '_' + self.timestep})

        for fgroup in fgroups:
            self._add_nested_ftype_filter(
                root_tree=root_tree,
                fgroup=fgroup,
                children=[parent + "__" + ft for ft in fgroups[fgroup]],
            )
        #                          children=[parent + '__' + ft + '_' + self.timestep for ft in fgroups[fgroup]])

        return None

    def _add_timeseries_filter(
        self,
        root_tree: ET.ElementTree,
        parent: str,
        children: list,
        childrenNames: list,
    ):
        """
        Creates new timeseries filter with contents for each child, constrained to parent

        Parameters
        ----------
        parent: str
            make the filter id: parent__child, name: childname; use for locationConstraints
        children: list
            a list containing all timeSeriesSet need to be added
        childrenNames: list
            correponsing names to param children
        """
        for child, childName in zip(children, childrenNames):
            child_tree = ET.SubElement(
                root_tree, "filter", {"id": parent + "__" + child, "name": childName}
            )
            xml_utils.add_child_with_text(
                child_tree, "timeSeriesSetsId", "all__" + child
            )
            constr_tree = ET.SubElement(child_tree, "locationConstraints")
            ET.SubElement(constr_tree, "idContains", {"contains": parent})
            # groupby_tree = ET.SubElement(child_tree, 'groupBy')
            # XmlUtil.add_child_with_text(groupby_tree, 'parameterAttributeId', 'ParameterGroup')
        return None

    def _create_filters_all(self) -> ET.ElementTree:
        """
        Creates new Filters_ribasim__all.xml

        Returns
        -------
        root_tree: ET.ElementTree

        * Note default filter is required, so set as all__RIB_QSW_day, change it when needed
        """

        # the feature and parameters
        param_dict = (
            self.param_dfs.dropna(subset=["FEATURE_TYPE", "INT_PARID"])
            .pivot_table(columns="FEATURE_TYPE", values=["INT_PARID"], aggfunc=list)
            .squeeze()
            .to_dict()
        )

        # start building the xml
        root_tree = xml_utils.create_fews_xml_root(schema="filters", version="1.1")

        # add default filter
        #        XmlUtil.add_child_with_text(root_tree, 'defaultFilterId', 'default')

        # add timeSeriesSets by feature type
        for feature in param_dict:
            timeSeriesSets_tree = ET.SubElement(
                root_tree, "timeSeriesSets", {"id": "all__" + feature}
            )
            #                timeSeriesSets_tree = ET.SubElement(root_tree, 'timeSeriesSets',
            #                                                    {'id': 'all__' + feature + '_' + self.timestep})

            # add timeSeriesSet by parameter
            for parameter in param_dict[feature]:
                self._add_timeseriesset(
                    root_tree=timeSeriesSets_tree,
                    feature=feature,
                    parameter=parameter,
                    timestep="nonequidistant",
                )  # self.timestep)

        return root_tree

    def add_default_filter(
        self, default_details: Dict, root_tree: ET.ElementTree
    ) -> ET.ElementTree:
        defaultfilter_tree = ET.SubElement(
            root_tree,
            "filter",
            {"id": default_details["id"], "name": default_details["name"]},
        )
        xml_utils.add_child_with_text(
            defaultfilter_tree, "timeSeriesSetsId", default_details["timeSeriesSetsId"]
        )

        return root_tree

    def _add_timeseriesset(
        self, root_tree: ET.ElementTree, feature: str, parameter: str, timestep: str
    ):
        """
        Add timeSeriesSet to the given timeSeriesSets, for each parameter of the network

        Parameters
        ----------
        root_tree:
            xml-root object - timeSeriesSets
        feature: str
            feature type
        parameter: str
            parameter type
        timestep: str
            timestep of current network
        """

        timeSeriesSet_tree = ET.SubElement(root_tree, "timeSeriesSet")
        xml_utils.add_child_with_text(
            timeSeriesSet_tree, "moduleInstanceSetId", "run_{}".format(self.codebase)
        )
        xml_utils.add_child_with_text(timeSeriesSet_tree, "valueType", "scalar")
        xml_utils.add_child_with_text(timeSeriesSet_tree, "parameterId", parameter)
        xml_utils.add_child_with_text(
            timeSeriesSet_tree, "locationSetId", "all_networks__" + feature
        )
        xml_utils.add_child_with_text(
            timeSeriesSet_tree, "timeSeriesType", "simulated forecasting"
        )
        ET.SubElement(timeSeriesSet_tree, "timeStep", {"id": timestep})
        xml_utils.add_child_with_text(
            timeSeriesSet_tree, "readWriteMode", "read complete forecast"
        )

    def create_network_filters_file(self, default_filter_details: Dict):
        """
        Main function to create network filters by feature for a specific model.

        Creates a Filters_ribasim__network.xml file with all time series set loaded (default filter all__RIB_QSW_day)
        Extends Filters_ribasim__all.xml with current network
        """
        root_tree = xml_utils.create_fews_xml_root(schema="filters", version="1.1")
        self._add_nested_fgroup_filter(
            root_tree=root_tree, parent=self.networkid, fgroups=self.fgroups_dict
        )
        self._add_timeseries_filter(
            root_tree=root_tree,
            parent=self.networkid,
            children=[f for f in self.features],
            childrenNames=self.featureNames,
        )
        #                    children = [f + '_' + self.timestep for f in self.features], childrenNames = self.featureNames)

        xml_utils.serialize_xml(
            root_tree,
            self.region_ntw_dir.joinpath(
                "Filters_{}__{}.xml".format(self.codebase, self.networkid)
            ),
        )

        filters_all_xml = self.region_rbs_dir.joinpath(
            "Filters_{}__all.xml".format(self.codebase)
        )
        #        filters_all_xml = self.region_rbs_dir.joinpath('Filters_{}__all_{}.xml'.format(self.codebase, self.timestep))
        if not filters_all_xml.exists():
            filters_all_tree = self._create_filters_all()

            filters_all_tree = self.add_default_filter(
                root_tree=filters_all_tree, default_details=default_filter_details
            )
            xml_utils.serialize_xml(filters_all_tree, filters_all_xml)


class SpatialDisplayXml:
    """
    Provides basic functions to create ribasim Spatial Displasy
    """

    def __init__(self, config: Dict, logger: logging.Logger):
        """
        initialisation

        Parameters
        ----------
        config: dict
            dict with at least the following keys:

                * networkid: str refereing the network /model
                * timestep: str timestepId as used in timesteps.xml
                * codebase: str codebase reference e.g ribasim, wflow
                * (optional) top, bottom, left, right: bounding box extent in WGS1984
                * cf_application_folder: application (Region_Home) of the Fews-application, holding the Config, Modules the Modules folder should hold: spatialdisplay_mapping.csv in subfolder <codebase>/system/cf_configurator/definitions/mappings
                * cf_config_folder: application/Config folder of the Fews-application

        """
        self.logger = logger

        self.ini = config
        # model and network name
        self.networkid = self.ini["networkid"]
        self.timestep = self.ini["timestep"]
        self.codebase = self.ini["codebase"]
        # extents
        self.includeBoundingBox = True if "top" in self.ini else False

        # Gridplotgroup folder structure and content
        spadispl_mappingfile = io.get_dir(
            self.ini["cf_application_folder"],
            subfolders=[
                "Modules",
                self.codebase,
                "system",
                "cf_configurator",
                "definitions",
                "mappings",
            ],
        ).joinpath(("spatialdisplay_mapping.csv"))
        if spadispl_mappingfile.exists():
            spadisp_dfs = pd.read_csv(spadispl_mappingfile, sep=";", header=0)
        else:
            msg = "Cannot find Spatial Display Mapping file: {}".format(
                spadispl_mappingfile
            )
            self.logger.error(msg)
            raise
        self.spadisp_dfs = spadisp_dfs.replace(
            r"\$NETWORKID\$", self.networkid, regex=True
        )
        # output directory
        self.display_dir = io.get_dir(
            self.ini["cf_config_folder"], subfolders=["DisplayConfigFiles"]
        )
        self.display_ntw_dir = io.get_dir(
            self.ini["cf_config_folder"],
            subfolders=["DisplayConfigFiles", self.networkid],
        )

        self.spadisp_template_xml = io.get_dir(
            self.ini["cf_application_folder"],
            subfolders=[
                "Modules",
                self.codebase,
                "system",
                "cf_configurator",
                "config_templates",
                "DisplayConfigFiles",
            ],
        ).joinpath("SpatialDisplay.xml")
        if not self.spadisp_template_xml.exists():
            self.logger.warning(
                "Cannot find Spatial Display template file, basic start: {}",
                self.spadisp_template_xml,
            )

        ET.register_namespace("", "http://www.wldelft.nl/fews")

    def __geomap__(self):
        """
        default geomap
        """
        geomap_str = """    
                    <geoMap>
                    <geoDatum>WGS 1984</geoDatum>
                    <projection>mercator</projection>
        			<defaultExtent id="Deltares">
				        <left>4.3</left>
		        		<right>4.45</right>
        				<top>52.03</top>
				        <bottom>51.97</bottom>
		        	</defaultExtent>
        			<extraExtent id="World">
				        <left>0</left>
		        		<right>100</right>
        				<top>0</top>
				        <bottom>45</bottom>
		        	</extraExtent>
                    <scaleBarVisible>true</scaleBarVisible>
                    <northArrowVisible>true</northArrowVisible>
                    <labelsVisible>true</labelsVisible>
                    <backgroundColor>light sky blue</backgroundColor>
                    <openStreetMapLayer id="Osm">
                        <url>http://tile.openstreetmap.org</url>
                        <visible>true</visible>
                        <cacheDir>%REGION_HOME%/MapCache/OsmTiles</cacheDir>
                    </openStreetMapLayer>
                    <layer id="Satellite">
                        <className>nl.wldelft.libx.openmap.GenericTileServerLayer</className>
                        <visible>false</visible>
                        <properties>
                            <string key="tileUrlPattern" value="http://h%RIGHT(QUAD_KEY,1)%.ortho.tiles.virtualearth.net/tiles/h%QUAD_KEY%.jpeg?g=1"/>
                            <string key="cacheDir" value="%REGION_HOME%/MapCache/Satellite_Cache"/>
                            <int key="minZoomLevel" value="2"/>
                            <int key="maxZoomLevel" value="17"/>
                            <int key="topZoomLevel" value="19"/>
                            <int key="tileSize" value="256"/>
                        </properties>
                    </layer>
                    <layerGroup id="{}" name="{} Networks">
                    </layerGroup>
                    </geoMap>"""
        geomap_str.format(self.codebase, self.codebase.title())
        geomap_tree = ET.fromstring(geomap_str)
        return geomap_tree

    def _add_timeseriesset(
        self,
        root_tree: ET.ElementTree,
        moduleInstanceSetId: str,
        parameterId: str,
        locationSetId: str,
        timeStep: str,
        valueType="scalar",
        timeSeriesType="simulated forecasting",
        readWriteMode="read complete forecast",
    ):
        """
        Add timeSeriesSet

        Parameters
        ----------
        root_tree:
            xml-root object - timeSeriesSets
        all others: string - elements
        """
        timeSeriesSet_tree = ET.SubElement(root_tree, "timeSeriesSet")
        xml_utils.add_child_with_text(
            timeSeriesSet_tree, "moduleInstanceSetId", moduleInstanceSetId
        )
        xml_utils.add_child_with_text(timeSeriesSet_tree, "valueType", valueType)
        xml_utils.add_child_with_text(timeSeriesSet_tree, "parameterId", parameterId)
        xml_utils.add_child_with_text(
            timeSeriesSet_tree, "locationSetId", locationSetId
        )
        xml_utils.add_child_with_text(
            timeSeriesSet_tree, "timeSeriesType", timeSeriesType
        )
        ET.SubElement(timeSeriesSet_tree, "timeStep", {"id": timeStep})
        xml_utils.add_child_with_text(
            timeSeriesSet_tree, "readWriteMode", readWriteMode
        )

    def _populate_gridPlotGroup(
        self, root_tree: ET.ElementTree, spadisp_dfs: pd.DataFrame
    ):
        """
        add gridPlotGroup based on spatialDisplay_mapping.csv

        Parameters
        ----------
        root_tree: ET.ElemnetTree
            gridDisplay_tree
        spadisp_dfs: pd.DataFrame
            spatialDisplay layout read from spatialDisplay_mapping.csv with current network information
        """

        #  add gridPlotGroup
        gridPlotGroup_id = spadisp_dfs.gridPlotGroup_id[0]
        gridPlotGroup_name = spadisp_dfs.gridPlotGroup_name[0]
        gridPlotGroup_tree = ET.SubElement(
            root_tree,
            "gridPlotGroup",
            {"id": gridPlotGroup_id, "name": gridPlotGroup_name},
        )

        #  extend gridPlotGroup
        for _, row in spadisp_dfs.iterrows():

            # add gridPlot
            gridPlot_tree = ET.SubElement(
                gridPlotGroup_tree,
                "gridPlot",
                {"id": row.gridPlot_id, "name": row.gridPlot_name},
            )

            # add dataLayer
            dataLayer_tree = ET.SubElement(gridPlot_tree, "dataLayer")
            xml_utils.add_child_with_text(
                dataLayer_tree, "showArrowsOnLines", str(row.showArrowsOnLines).lower()
            )

            # add timeseriesSet
            self._add_timeseriesset(
                root_tree=dataLayer_tree,
                moduleInstanceSetId=row.moduleInstanceSetId,
                parameterId=row.parameterId,
                locationSetId=row.locationSetId,
                timeStep=self.timestep,
            )

            # add others
            xml_utils.add_child_with_text(
                gridPlot_tree, "classBreaksId", row.classBreaksId
            )

    def _create_spatialdisplay_xml(self, out_file: Path):
        """
        Create SpatialDisplay.xml file from scratch

        Parameters
        ----------
        out_file: Path
            Path output file to write

        """
        # create SpatialDisplay_ribasim__networkid.xml
        root_tree = xml_utils.create_fews_xml_root(schema="gridDisplay")
        # add title
        ET.SubElement(root_tree, "title")
        # add defaults
        ET.SubElement(root_tree, "defaults").append(self.__geomap__())
        # write
        xml_utils.serialize_xml(root_tree, out_file)
        return None

    def add_extra_extent(self, xml_root: ET.ElementTree):
        """
        Adds extra extent to geomap (after default extent) if model/networkid is not yet defined

        Parameters
        ----------
        xml_root: ET.ElementTree
            Spatial Display root
        """

        def _get_index(root, element):
            for idx, child in enumerate(root):
                if element == child:
                    return idx
            else:
                raise ValueError(
                    "No '%s' tag found in '%s' children" % (element.tag, root.tag)
                )

        def _new_extra_extent():
            extra_tree = ET.Element("extraExtent", {"id": self.networkid})
            xml_utils.add_child_with_text(extra_tree, "left", self.ini["left"])
            xml_utils.add_child_with_text(extra_tree, "right", self.ini["right"])
            xml_utils.add_child_with_text(extra_tree, "top", self.ini["top"])
            xml_utils.add_child_with_text(extra_tree, "bottom", self.ini["bottom"])

            return extra_tree

        for default in xml_root.iter("{http://www.wldelft.nl/fews}defaults"):
            geomap = default.find("{http://www.wldelft.nl/fews}geoMap")
            if geomap is not None:
                extra_extents = geomap.findall(
                    "{http://www.wldelft.nl/fews}extraExtent"
                )
                if extra_extents is not None:
                    existing_extents = [
                        extra_extent.attrib["id"] for extra_extent in extra_extents
                    ]
                    if self.networkid in existing_extents:
                        break
                elem_index = _get_index(
                    geomap, geomap.find("{http://www.wldelft.nl/fews}defaultExtent")
                )
                geomap.insert(elem_index + 1, _new_extra_extent())

        return None

    def add_child_layer_group(self, xml_root: ET.ElementTree, shape_files: List):
        """
        Function to add network child layer group to a parent layer group

        Parameters
        ----------
        xml_root:
            spatial display
        parentgrp: str
            parent group where child layers are added
        """

        def _find_layergroup(xml_root: ET.ElementTree):
            # find geomap with layer group to expand
            for default in xml_root.iter("{http://www.wldelft.nl/fews}defaults"):
                geomap = default.find("{http://www.wldelft.nl/fews}geoMap")
                if len(geomap) > 0:
                    toplayergroups = geomap.findall(
                        "{http://www.wldelft.nl/fews}layerGroup"
                    )
                    if len(toplayergroups) > 0:
                        for toplayergroup in toplayergroups:
                            if toplayergroup.attrib["id"].lower() == self.codebase:
                                return toplayergroup
                    return ET.SubElement(
                        geomap,
                        "layerGroup",
                        {"id": self.codebase, "name": self.codebase.title()},
                    )
                self.logger.warning("No GeoMap available, cannot add layergroup")
            return None

        def _add_network_layergroup(layergroups: ET.Element, shape_files: List):
            """
            Add shape files to layergroudps element of this codebase

            Parameters
            ----------
            layergroups: ET.Element
                layergroups element to expand
            shape_files :list
                List of dicts referencing shapefiles to include. Dict should hold id, file (name), pointSize (optional for nodes), lineColor (optional for links)
            """
            networklayergroup = ET.SubElement(
                layergroups, "layerGroup", {"id": self.networkid}
            )
            for shape in shape_files:
                layer = ET.SubElement(
                    networklayergroup, "esriShapeLayer", {"id": shape["id"]}
                )
                xml_utils.add_child_with_text(layer, "file", shape["file"])
                xml_utils.add_child_with_text(layer, "geoDatum", "WGS 1984")
                xml_utils.add_child_with_text(layer, "visible", "true")
                if "pointSize" in shape:
                    xml_utils.add_child_with_text(
                        layer, "pointSize", shape["pointSize"]
                    )
                elif "lineColor" in shape:
                    xml_utils.add_child_with_text(
                        layer, "lineColor", shape["lineColor"]
                    )

        toplayergroup = _find_layergroup(xml_root)
        if toplayergroup is None:
            _add_network_layergroup(layergroups=toplayergroup, shape_files=shape_files)
        else:
            layergroups = toplayergroup.findall(
                "{http://www.wldelft.nl/fews}layerGroup"
            )
            existing_layergroups = [
                layergroup.attrib["id"] for layergroup in layergroups
            ]
        if not self.networkid in existing_layergroups:
            _add_network_layergroup(layergroups=toplayergroup, shape_files=shape_files)
        # resulting_layergroups = [layergroup.attrib['id'] for layergroup in layergroups]

    def _append_gridplotgroup(self, xml_root: ET.ElementTree):
        """
        Appends gridplotgroups to the SpatialDisplay.xml

        Parameters
        ----------
        xml_root: ET.ElementTree
        """
        gridplotgroups = xml_root.findall("{http://www.wldelft.nl/fews}gridPlotGroupId")
        gridplotgroupids = [gridplotgroup.text for gridplotgroup in gridplotgroups]
        if (
            not "{}__{}_results".format(self.networkid, self.codebase)
            in gridplotgroupids
        ):
            xml_utils.add_child_with_text(
                xml_root,
                "gridPlotGroupId",
                "{}__{}_results".format(self.networkid, self.codebase),
            )
        # resulting_gridplotgroups = [gridplotgroup.text for gridplotgroup in gridplotgroups]

    def _append_spatialdisplay_xml(self, shapefiles: List):
        """
        Main function to append network to SpatialDisplay file (extent, layergroup and plotgroup)

        Parameters
        ----------
        shape_files: List of dicts referencing shapefiles to include. Dict should hold id, file (name), pointSize (optional for nodes), lineColor (optional for links)
        """

        fname = self.display_dir.joinpath("SpatialDisplay.xml")

        try:
            tree = ET.parse(fname)
            xml_root = tree.getroot()
        except:
            if self.spadisp_template_xml.exists():
                tree = ET.parse(self.spadisp_template_xml)
                xml_root = tree.getroot()
            else:
                self._create_spatialdisplay_xml(out_file=fname)
        if self.includeBoundingBox:
            self.add_extra_extent(xml_root=xml_root)
        self.add_child_layer_group(xml_root=xml_root, shape_files=shapefiles)
        self._append_gridplotgroup(xml_root=xml_root)
        # write
        xml_utils.serialize_xml(
            xml_root, self.display_dir.joinpath("SpatialDisplay.xml")
        )

    def create_network_spatialplots_file(self, shape_files: List):
        """
        Function to create SpatialPlots file with gridplotgroups for a specific network

        Generates one file per network
        Calls append for SpatialDisplay file

        Parameters
        ----------
        shape_files: list
            List of dicts referencing shapefiles to include. Dict should hold id, file (name), pointSize (optional for nodes), lineColor (optional for links)
        """

        # create SpatialPlots_ribasim__networkid.xml
        root_tree = xml_utils.create_fews_xml_root(schema="gridPlotGroups")

        # add gridPlotGroup
        self._populate_gridPlotGroup(root_tree=root_tree, spadisp_dfs=self.spadisp_dfs)

        # write
        xml_utils.serialize_xml(
            root_tree,
            self.display_ntw_dir.joinpath(
                "SpatialPlots_{}__{}.xml".format(self.codebase, self.networkid)
            ),
        )
        self._append_spatialdisplay_xml(shapefiles=shape_files)
        return None


# TODO: generalize and remove RIBASIM specific stuff
class LocationSetsXml:
    """
    Provides basic functions to create network locationsets
    """

    def __init__(self, config: Dict, logger: logging.Logger):
        # config file
        self.logger = logger

        self.ini = config
        self.codebase = self.ini["codebase"]
        self.codeshort = (
            self.codebase[0:3].upper()
            if len(self.codebase) > 4
            else self.codebase.upper()
        )

        #        TODO: RIBASIM SPECIFIC ONLY ???
        #        # list of featuretypes
        #        self.ftype_names = list(self.ini['featuretypes_df'].ftype)
        #        # list of links and node groups (subset of feature types)
        #
        #        self.linktypes = list(self.ini['featuretypes_df'].dropna(subset=['link_color']).ftype)
        #        self.output_ftypes = list(self.ini['featuretypes_df'].dropna(subset=['ftype_group']).ftype)
        #
        #        #feature type definitions
        #        self.ftype_defs = FeatureTypeDefinitions(config=config, logger=self.logger,
        #                                               featuretypes=self.ftype_names)
        #        self.ftype_defs.get_definitions()

        # model and networkid
        self.networkid = self.ini["networkid"]
        self.timestep = self.ini["timestep"]
        self.codebase = self.ini["codebase"]
        self.base_templates_dir = io.get_dir(
            self.ini["cf_application_folder"],
            subfolders=[
                "Modules",
                self.codebase,
                "system",
                "cf_configurator",
                "config_templates",
            ],
        )
        self.system_cfg_dir = io.get_dir(
            self.ini["cf_config_folder"], subfolders=["SystemConfigFiles"]
        )
        self.display_cfg_dir = io.get_dir(
            self.ini["cf_config_folder"], subfolders=["DisplayConfigFiles"]
        )
        # RegionConfigFiles folder and subfolders for codebase general and network
        self.region_cb_dir = io.get_dir(
            self.ini["cf_config_folder"],
            subfolders=["RegionConfigFiles", self.codebase, "general"],
        )
        self.region_ntw_dir = io.get_dir(
            self.ini["cf_config_folder"],
            subfolders=["RegionConfigFiles", self.networkid],
        )

        self.prep_shape_dir = io.get_dir(
            self.ini["newmodel_folder"], subfolders=["network", "shapefiles"]
        )
        self.prep_csvdata_dir = io.get_dir(
            self.ini["newmodel_folder"], subfolders=["network", "data"]
        )
        self.maplayer_ntw_dir = io.get_dir(
            self.ini["cf_config_folder"], subfolders=["MapLayerFiles", self.networkid]
        )

        ET.register_namespace("", "http://www.wldelft.nl/fews")

    def copy_network_shape_to_maplayers(self):

        shpfiles = [x for x in self.prep_shape_dir.iterdir() if x.is_file()]
        for srcFile in shpfiles:
            if srcFile.suffix.lower() in [".dbf", ".shp", ".shx", ".prj"]:
                tgtFile = self.maplayer_ntw_dir.joinpath(
                    "{}__{}_{}".format(self.networkid, self.codeshort, srcFile.name)
                )
                shutil.copy2(srcFile, tgtFile)
            elif ".ntw" in srcFile.suffix.lower():
                continue
            else:
                logging.debug("Unidentified file type, skipping file %s" % srcFile)

    def copy_csvdata_to_maplayers(self):

        csvdatafiles = [x for x in self.prep_csvdata_dir.iterdir() if x.is_file()]

        ftype_names = [x[:7] for x in self.ftype_names]
        for srcFile in csvdatafiles:
            if srcFile.stem[:7] in ftype_names:
                tgtFile = self.maplayer_ntw_dir.joinpath(
                    self.networkid + "__" + srcFile.name
                )
                shutil.copy2(srcFile, tgtFile)
            else:
                logging.debug("Unidentified feature type, skipping file %s" % srcFile)

    def _add_attributefiles(
        self,
        xml_tree: ET.ElementTree,
        ftype: str,
        matchingId: str,
        inclPrefixInAttrId: bool,
    ):
        """
        Add attribute files for each feature type to the locationset

        Parameters
        ----------
        xml_tree: ET.ElementTree
            xml-object to expand
        ftype: str
        matchingId: str
        inclPrefixInAttrId: bool
        """

        def _add_attribute_file(
            self,
            xml_tree: ET.ElementTree,
            csvfile: str,
            matchingId: str,
            ftype: str,
            ft_defs: pd.DataFrame,
            field: str,
            inclPrefixInAttrId: bool,
        ):
            """
            Add attribute-file to locationset for a specific feature type

            Parameters
            ----------
            xml_tree: ET.ElementTree
                location set to expand
            ftype: str
                feature type to add
            ft_defs: pd.DataFrame
                dataframes, associated feature type definition as defined in CSV-file
            """

            def _add_property(
                tree: ET.ElementTree,
                prop: pd.DataFrame,
                item: str,
                inclPrefixInAttrId: bool,
            ):
                """
                Add property/attribute to xml-element

                Parameters
                ----------
                tree: ET.ElementTree
                    attribute-element to expand
                prop: pd.DataFrame
                    dataframe holding all columnvalues from CSV-definition
                item: str
                    PropertyName or ColumnName
                """
                name = prop[item]
                if inclPrefixInAttrId:
                    id = "{}_{}".format(prop.FeatureType, prop[item].replace(" ", "_"))
                else:
                    id = prop[item]

                # check %-clashes with Fews-config
                if "%" in name:
                    raise ValueError("Found % in propertyname {}".format(name))
                if isinstance(prop.DataType, str):
                    datatype = prop.DataType.lower()
                else:
                    raise ValueError(
                        "Found incorrect or no datatype in propertyname {}".format(name)
                    )
                attr_tree = ET.SubElement(
                    tree, "attribute", {"id": id, "name": prop.GUI_caption_en}
                )
                if item == "ColumnName":
                    xml_utils.add_child_with_text(attr_tree, "text", "%" + name + "%")
                elif datatype in ["boolean", "bool"]:
                    xml_utils.add_child_with_text(
                        attr_tree, "boolean", "%" + name + "%"
                    )
                elif (
                    datatype == "long" and name[-5:].lower() == "index"
                ):  # TMS or lookup index
                    xml_utils.add_child_with_text(attr_tree, "text", "%" + name + "%")
                elif datatype in ["long", "single", "double", "featureid"]:
                    xml_utils.add_child_with_text(attr_tree, "number", "%" + name + "%")
                elif datatype in ["string", "list", "listprovider", "enum"]:
                    xml_utils.add_child_with_text(attr_tree, "text", "%" + name + "%")
                else:
                    raise ValueError(
                        "Found incorrect datatype in propertyname {}".format(name)
                    )

            af_tree = ET.SubElement(xml_tree, "attributeFile")
            xml_utils.add_child_with_text(af_tree, "csvFile", csvfile)
            xml_utils.add_child_with_text(af_tree, "id", matchingId)
            try:
                # walk over all property definitions, series definitions and table definitions for this feature type
                for i, row in ft_defs.iterrows():
                    _add_property(
                        tree=af_tree,
                        prop=row,
                        item=field,
                        inclPrefixInAttrId=inclPrefixInAttrId,
                    )
            except ValueError as e:
                msg = "{} for {}".format(e, ftype)
                self.logger.error(msg)

    #        # RIBASIM SPECIFIC ??
    #        ftype_props, ftype_series, ftype_tables = self.ftype_defs.get_relevant_cf_attributes_by_ftype(featuretype=ftype)
    #        if ftype_props.index.size > 0:
    #            fname = self.networkid + '__' + ftype + '_properties.csv'
    #            _add_attribute_file(self, xml_tree=xml_tree, ftype=ftype, csvfile=fname, matchingId=matchingId,
    #                    ft_defs=ftype_props, field='PropertyName',inclPrefixInAttrId=inclPrefixInAttrId)
    #        if ftype_series.index.size > 0:
    #            unique_series = list(ftype_series['SeriesColumnDefinitions'].drop_duplicates())
    #            for s in unique_series:
    #                series_df = ftype_series[ftype_series['SeriesColumnDefinitions'] == s]
    #                fname = self.networkid + '__' + ftype + '_series_'+ s + '.csv'
    #                _add_attribute_file(self, xml_tree=xml_tree, ftype=ftype, csvfile=fname, matchingId=matchingId,
    #                    ft_defs=series_df, field='ColumnName', inclPrefixInAttrId=inclPrefixInAttrId)
    #        if ftype_tables.index.size > 0:
    #            unique_tables = list(ftype_tables['TableColumnDefinitions'].drop_duplicates())
    #            for t in unique_tables:
    #                series_df = ftype_tables[ftype_tables['TableColumnDefinitions'] == t]
    #                fname = self.networkid + '__' + ftype + '_table_'+ t + '.csv'
    #                _add_attribute_file(self, xml_tree=xml_tree, ftype=ftype, csvfile=fname, matchingId=matchingId,
    #                    ft_defs=series_df, field='ColumnName', inclPrefixInAttrId=inclPrefixInAttrId)

    def _copy_locationicons(self):
        """
        Copies template file: LocationIcons
        """
        srcFile = self.base_templates_dir.joinpath(
            "SystemConfigFiles", "LocationIcons.xml"
        )
        tgtFile = self.system_cfg_dir.joinpath("LocationIcons.xml")
        if not tgtFile.exists():
            if not srcFile.exists():
                msg = "Cannot find template file: {} for copy".format(srcFile)
                self.logger.error(msg)
            else:
                shutil.copy2(srcFile, tgtFile)

    def _copy_modifiertypes(self):
        """
        Copies template file: ModifierTypes
        """
        srcFile = self.base_templates_dir.joinpath(
            "RegionConfigFiles", "ModifierTypes.xml"
        )
        tgtFile = self.region_rbs_dir.parent.joinpath("ModifierTypes.xml")
        if not srcFile.exists():
            msg = "Cannot find template file: {} for copy".format(srcFile)
            self.logger.error(msg)
        if not tgtFile.exists():
            shutil.copy2(srcFile, tgtFile)

    def _copy_modifiersdisplay(self):
        """
        Copies template file: ModifiersDisplay
        """
        srcFile = self.base_templates_dir.joinpath(
            "DisplayConfigFiles", "ModifiersDisplay.xml"
        )
        tgtFile = self.display_cfg_dir.joinpath("ModifierDisplay.xml")
        if not srcFile.exists():
            msg = "Cannot find template file: {} for copy".format(srcFile)
            self.logger.error(msg)
        if not tgtFile.exists():
            shutil.copy2(srcFile, tgtFile)
