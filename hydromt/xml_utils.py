#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic xml related convience functions."""
import xml.etree.ElementTree as ET


def add_child_with_text(parent, child, text):
    """Helper function to create a xml child element with text"""
    child = ET.SubElement(parent, child)
    child.text = text


def create_xml_root(ns, schema_url, schema_name, extra_attributes={}):
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


def create_xml_parent(ns, schema_url, schema_name, extra_attributes={}):
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


def create_fews_xml_root(schema: str, version: str = None) -> ET.ElementTree:
    """
    Function to create xml rot with fews namespace and url.
    
    Arguments
    ----------
    schema: str
        schema to create
    version: str, Optional
        version to create, default = 1.1
    Returns
    -------
    xml_root element: ET.element
    """
    _ns = {
        "xmlns": "http://www.wldelft.nl/fews",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }
    _url = "http://fews.wldelft.nl/schemas/version1.0/"
    if version is None:
        return create_xml_root(ns=_ns, schema_url=_url, schema_name=schema)
    else:
        return create_xml_root(
            ns=_ns,
            schema_url=_url,
            schema_name=schema,
            extra_attributes={"version": version},
        )


def serialize_xml(root, file_name):
    """Helper function to serialize xml"""
    ET.ElementTree(root).write(file_name, encoding="UTF-8", xml_declaration=True)
