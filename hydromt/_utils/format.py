"""Some formatting objects and functions."""

import yaml


# I/O related
class CatalogDumper(yaml.SafeDumper):
    """Custom class for writing style of yaml files."""

    def write_line_break(self, data=None):
        """Write data with linebreaks between first level headers."""
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()
