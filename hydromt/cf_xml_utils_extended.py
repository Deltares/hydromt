import xml.etree.ElementTree as ET
import logging
from typing import List, Set, Dict, Tuple, Optional, Callable
from pathlib import Path


class CfXmlUtilsExtended:
    """
    Provides basic functions to populate and insert fews-xml-snippets
    """

    def __init__(self, logger: logging.Logger):
        """
        initialisation
        :param: log_name - name as appears in logger
        """
        self.logger = logger

        ET.register_namespace("", "http://www.wldelft.nl/fews")

    def _get_index(self, root, element):
        for idx, child in enumerate(root):
            if element == child:
                return idx
        else:
            raise ValueError(
                "No '%s' tag found in '%s' children" % (element.tag, root.tag)
            )

    def _new_extra_extent(
        self, region: str, top: str, bottom: str, left: str, right: str
    ):
        extra_tree = ET.Element("extraExtent", {"id": region})
        self._add_child_with_text(extra_tree, "left", left)
        self._add_child_with_text(extra_tree, "right", right)
        self._add_child_with_text(extra_tree, "top", top)
        self._add_child_with_text(extra_tree, "bottom", bottom)
        return extra_tree

    def get_xml_root(self, xml_file: str, xsd_schema_name: str):
        """
        Obtains xml-root object, checks for file exist and schema
        :param xml_file:
        :param xsd_schema_name:
        :return:
        """
        assert Path(xml_file).is_file(), f"Cannot find file {xml_file}"

        try:
            tree = ET.parse(xml_file)
            xml_root = tree.getroot()
        except Exception as e:
            msg = f"Parsing xml file {xml_file} raised Exception {e}"
            self.logger.error(msg)
            raise msg
        has_schema = xml_root.tag == "{http://www.wldelft.nl/fews}%s" % xsd_schema_name
        assert (
            has_schema
        ), f"Schema of {xml_file} not as expected {xsd_schema_name}.xsd)"
        return xml_root

    def _add_child_with_text(self, parent, child, text):
        """Helper function to create a xml child element with text"""

        child = ET.SubElement(parent, child)
        child.text = text

    def _create_xml_root(self, ns, schema_url, schema_name, extra_attributes={}):
        """Helper function to create xml root"""
        attributes = {
            "xmlns:xsi": ns["xmlns:xsi"],
            "xsi:schemaLocation": "{} {}{}.xsd".format(
                ns["xmlns"], schema_url, schema_name
            ),
        }
        attributes.update(extra_attributes)

        ET.register_namespace("", ns["xmlns"])
        ET.register_namespace("xsi", ns["xmlns:xsi"])
        root = ET.Element("{{{}}}{}".format(ns["xmlns"], schema_name), attributes)

        return root

    def _create_xml_parent(self, ns, schema_url, schema_name, extra_attributes={}):
        """Helper function to create xml root"""
        attributes = {
            "xmlns:xsi": ns["xmlns:xsi"],
            "xsi:schemaLocation": "{} {}{}.xsd".format(
                ns["xmlns"], schema_url, schema_name
            ),
        }
        attributes.update(extra_attributes)

        ET.register_namespace("", ns["xmlns"])
        ET.register_namespace("xsi", ns["xmlns:xsi"])
        root = ET.Element("{{{}}}{}".format(ns["xmlns"], schema_name), attributes)

        return root

    def get_ids_from_xml(self, xml_file, element) -> list:
        """
        get a list of ids from each element

        Parameters
        ----------
        xml_file: str
            The xml file to parse.
        element: str
            name of the element whoes id will be found
        """
        tree = ET.parse(Path(xml_file))
        return [e.attrib["id"] for e in tree.findall(""".//{*}%s""" % (element))]

    def _add_network_layergroup(
        self, layergroups: ET.Element, shape_files: List, region: str
    ):
        """
        Add shape files to layergroups element of this codebase
        :param layergroups: layergroups element to expand
        :param shape_files: List of dicts referencing shapefiles to include. Dict should hold id, file (name),
        pointSize (optional for nodes), lineColor (optional for links)
         :return:
        """
        layergroup = ET.SubElement(layergroups, "layerGroup", {"id": region})
        for shape in shape_files:
            layer = ET.SubElement(layergroup, "esriShapeLayer", {"id": shape["id"]})
            self._add_child_with_text(layer, "file", shape["file"])
            self._add_child_with_text(layer, "geoDatum", "WGS 1984")
            self._add_child_with_text(layer, "visible", "true")
            if "pointSize" in shape:
                self._add_child_with_text(layer, "pointSize", shape["pointSize"])
            elif "lineColor" in shape:
                self._add_child_with_text(layer, "lineColor", shape["lineColor"])
        return layergroup

    def _add_geojson_layergroup(
        self, layergroups: ET.Element, geojson_files: List, region: str
    ):
        """
        Add geojson files to layergrouds element of this codebase
        :param layergroups: layergroups element to expand
        :param geojsone_files: List of dicts referencing geojsonfiles to include. Dict should hold id, file (name),
        shapetype
         :return:
        """
        layergroup = ET.SubElement(layergroups, "layerGroup", {"id": region})
        for geojson in geojson_files:
            assert geojson["shapeType"] in [
                "point",
                "line",
                "polygon",
            ], "shapeType should be point, line or polygon"

            layer = ET.SubElement(layergroup, "geoJsonLayer", {"id": geojson["id"]})
            self._add_child_with_text(layer, "file", geojson["file"])
            self._add_child_with_text(layer, "shapeType", geojson["shapeType"])
            self._add_child_with_text(layer, "geoDatum", "WGS 1984")
            self._add_child_with_text(layer, "visible", "true")
            if "lineColor" in geojson:
                self._add_child_with_text(layer, "lineColor", geojson["lineColor"])
            if "fillColor" in geojson:
                self._add_child_with_text(layer, "fillColor", geojson["fillColor"])
            if "pointSize" in geojson:
                self._add_child_with_text(layer, "pointSize", geojson["pointSize"])
        return layergroup

    def _add_layergroup_to_toplayer(
        self, toplayergroup: ET.Element, geometries: List, region: str, codebase: str
    ):
        """
        Adds layergroup to toplayergroup, geometries can refer to shape-files or geojason files
        :param layergroups: layergroups element to expand
        :param shape_files: List of dicts referencing shapefiles to include. Dict should hold id, file (name),
        pointSize (optional for nodes), lineColor (optional for links)
         :return:
        """
        if toplayergroup is None:
            existing_layergroups = []
        else:
            layergroups = toplayergroup.findall(
                "{http://www.wldelft.nl/fews}layerGroup"
            )
            existing_layergroups = [
                layergroup.attrib["id"] for layergroup in layergroups
            ]

        if not region in existing_layergroups:
            if codebase.lower() == "ribasim":
                toplayergroup = self._add_network_layergroup(
                    layergroups=toplayergroup, shape_files=geometries, region=region
                )
            elif codebase.lower() == "wflow":
                toplayergroup = self._add_geojson_layergroup(
                    layergroups=toplayergroup, geojson_files=geometries, region=region
                )
            else:
                self.logger.warning(f"Codebase {codebase} not supported yet")
        return toplayergroup

    def create_fews_xml_root(self, schema: str, version: str = None) -> ET.ElementTree:
        """
        Function to create xml root with fews namespace and url
        :param schema: schema to create
        :param version: version to create, default = 1.1
        :return: xml_root element
        """
        _ns = {
            "xmlns": "http://www.wldelft.nl/fews",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }
        _url = "http://fews.wldelft.nl/schemas/version1.0/"
        if version is None:
            return self._create_xml_root(ns=_ns, schema_url=_url, schema_name=schema)
        else:
            return self._create_xml_root(
                ns=_ns,
                schema_url=_url,
                schema_name=schema,
                extra_attributes={"version": version},
            )

    def serialize_xml(self, root, file_name):
        """Helper function to serialize xml"""
        ET.ElementTree(root).write(file_name, encoding="UTF-8", xml_declaration=True)

    def append_group_to_topology_xml(self, topology_file: str, groupId: str):
        """
        Adds topologygroup to topology-xml file
        :param topology_file: str Topology.xml file
        :return:
        """
        xml_root = self.get_xml_root(xml_file=topology_file, xsd_schema_name="topology")

        # find scheme
        inserts = 0
        groups = xml_root.findall("{http://www.wldelft.nl/fews}groupId")
        groupids = [group.text for group in groups]
        if not f"{groupId}" in groupids:
            self._add_child_with_text(xml_root, "groupId", f"{groupId}")
            inserts += 1
        if inserts > 0:
            self.serialize_xml(xml_root, topology_file)
            return True
        return False

    def append_group_to_topology_xml_object(
        self, xml_root: ET.Element, groupId: str, groupName: str
    ):
        """
        Adds topologygroup to topology-xml-object
        :return:
        """
        try:
            # find scheme
            nodes_tree = [
                n
                for n in xml_root.findall("{http://www.wldelft.nl/fews}nodes")
                if n.attrib["id"] == f"{groupId}"
            ][0]
        except:
            # add scheme
            nodes_tree = ET.SubElement(
                xml_root, "nodes", {"id": groupId, "name": groupName}
            )
        return xml_root

    def append_group_to_topologygroup_xml_object(
        self, xml_root: ET.ElementTree, parentgroupId: str, childgroupId: str
    ):
        """
        adds topologygroup to topology group object
        :return: xml-object
        """
        try:
            nodes_tree = [
                n
                for n in xml_root.findall("{http://www.wldelft.nl/fews}nodes")
                if n.attrib["id"] == f"{parentgroupId}"
            ][0]
        except Exception as e:
            msg = f"Searching parent group {parentgroupId} raised Exception {e}"
            self.logger.error(msg)
            raise msg
        grps = [
            g
            for g in nodes_tree.findall("{http://www.wldelft.nl/fews}groupId")
            if g.text == childgroupId
        ]
        if len(grps) < 1:
            self._add_child_with_text(nodes_tree, "groupId", childgroupId)
        return xml_root

    def insert_extra_extent_into_explorer_xml(
        self,
        explorer_file: str,
        region: str,
        top: float,
        bottom: float,
        left: float,
        right: float,
    ):
        """
        Function to add extra extent to map (after default extent) if region is not yet defined
        :param explorer_file =  filename, str to open, extend and save
        :param region =  region to add
        :param top =  top of extent in default geodatum as configured in Explorer
        :param bottom =  bottom of extent  in default geodatum as configured in Explorer
        :param left =  left of extent  in default geodatum as configured in Explorer
        :param right =  right of extent  in default geodatum as configured in Explorer
        :return: xml-object
        """

        xml_root = self.get_xml_root(xml_file=explorer_file, xsd_schema_name="explorer")

        geomap = xml_root.find("{http://www.wldelft.nl/fews}map")
        if len(geomap) > 0:
            extra_extents = geomap.findall("{http://www.wldelft.nl/fews}extraExtent")
            if extra_extents is not None:
                existing_extents = [
                    extra_extent.attrib["id"] for extra_extent in extra_extents
                ]
                if not region in existing_extents:
                    elem_index = self._get_index(
                        geomap, geomap.find("{http://www.wldelft.nl/fews}defaultExtent")
                    )
                    geomap.insert(
                        elem_index + 1,
                        self._new_extra_extent(
                            region,
                            "%.3f" % (top),
                            "%.3f" % (bottom),
                            "%.3f" % (left),
                            "%.3f" % (right),
                        ),
                    )

        self.serialize_xml(xml_root, explorer_file)
        return None

    def insert_extra_extent_into_spatial_display_xml(
        self,
        spatial_display_file: str,
        region: str,
        top: float,
        bottom: float,
        left: float,
        right: float,
    ):
        """
        Function to add extra extent to geomap (after default extent) if region is not yet defined
        :param fname =  filename, str to open, extend and save
        :param region =  region to add
        :param top =  top of extent in default geodatum as configured in SpatialDisplay
        :param bottom =  bottom of extent  in default geodatum as configured in SpatialDisplay
        :param left =  left of extent  in default geodatum as configured in SpatialDisplay
        :param right =  right of extent  in default geodatum as configured in SpatialDisplay
        :return: xml-object
        """

        xml_root = self.get_xml_root(
            xml_file=spatial_display_file, xsd_schema_name="gridDisplay"
        )

        inserts = 0
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
                    if not region in existing_extents:
                        elem_index = self._get_index(
                            geomap,
                            geomap.find("{http://www.wldelft.nl/fews}defaultExtent"),
                        )
                        geomap.insert(
                            elem_index + 1,
                            self._new_extra_extent(
                                region,
                                "%.3f" % (top),
                                "%.3f" % (bottom),
                                "%.3f" % (left),
                                "%.3f" % (right),
                            ),
                        )
                        inserts += 1

        if inserts > 0:
            self.serialize_xml(xml_root, spatial_display_file)
            return True
        return xml_root

    def append_gridplotgroup_to_spatial_display_xml(
        self,
        spatial_display_file: str,
        groupId: str,
    ):
        """
        Appends gridplotgroups to the SpatialDisplay.xml
        :param xml_root:
        :return:
        """
        xml_root = self.get_xml_root(
            xml_file=spatial_display_file, xsd_schema_name="gridDisplay"
        )

        inserts = 0
        gridplotgroups = xml_root.findall("{http://www.wldelft.nl/fews}gridPlotGroupId")
        gridplotgroupids = [gridplotgroup.text for gridplotgroup in gridplotgroups]
        if not f"{groupId}" in gridplotgroupids:
            self._add_child_with_text(xml_root, "gridPlotGroupId", f"{groupId}")
            inserts += 1
        if inserts > 0:
            self.serialize_xml(xml_root, spatial_display_file)
            return True
        return False

    def add_geometries_to_defaults_spatial_display_xml_object(
        self, xml_root: ET.ElementTree, geometries: List, region: str, codebase: str
    ):
        """
        Function to add network child layer group to a parent layer group
        :param xml_root: spatial display xml object
        :param shapes: list of shapes to add (either esri shp or geojson)
        :param region: region distinguishes sub-layergroup
        :param codebase: codebase distinguishes top layergroup
        :return: xml-object
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
                            if toplayergroup.attrib["id"].lower() == codebase:
                                return toplayergroup
                    return ET.SubElement(
                        geomap, "layerGroup", {"id": codebase, "name": codebase.title()}
                    )
                self.logger.warning("No GeoMap available, cannot add layergroup")
            return None

        toplayergroup = _find_layergroup(xml_root)
        toplayergroup = self._add_layergroup_to_toplayer(
            toplayergroup=toplayergroup,
            geometries=geometries,
            region=region,
            codebase=codebase,
        )

        return xml_root

    def add_geometries_to_explorer_xml_object(
        self, xml_root: ET.ElementTree, geometries: List, region: str, codebase: str
    ):
        """
        Function to add network child layer group to a parent layer group
        :param xml_root: explorer xml-object
        :param shapes: list of shapes to add
        :param region: region istinguishes sub-layergroup
        :param codebase: codebase distinguishes top layergroup
        :return: xml-object
        """

        def _find_layergroup(xml_root: ET.ElementTree):
            """
            find layer group in geomap referencing code base to expand
            :param xml_root:
            :return:
            """
            geomap = xml_root.find("{http://www.wldelft.nl/fews}map")
            if len(geomap) > 0:
                toplayergroups = geomap.findall(
                    "{http://www.wldelft.nl/fews}layerGroup"
                )
                if len(toplayergroups) > 0:
                    for toplayergroup in toplayergroups:
                        if toplayergroup.attrib["id"].lower() == codebase:
                            return toplayergroup
                return ET.SubElement(
                    geomap, "layerGroup", {"id": codebase, "name": codebase.title()}
                )
            self.logger.warning("No GeoMap available, cannot add layergroup")
            return None

        toplayergroup = _find_layergroup(xml_root)
        toplayergroup = self._add_layergroup_to_toplayer(
            toplayergroup=toplayergroup,
            geometries=geometries,
            region=region,
            codebase=codebase,
        )

        return xml_root
