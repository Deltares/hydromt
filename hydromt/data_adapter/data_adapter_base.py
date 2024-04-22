"""BaseModel for DataAdapter."""

from datetime import timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from hydromt._typing import TimeRange


class DataAdapterBase(BaseModel):
    """BaseModel for DataAdapter."""

    model_config = ConfigDict(extra="forbid")

    unit_add: Dict[str, Any] = Field(default_factory=dict)
    unit_mult: Dict[str, Any] = Field(default_factory=dict)
    rename: Dict[str, str] = Field(default_factory=dict)

    def to_source_timerange(
        self,
        time_range: TimeRange,
    ) -> TimeRange:
        """
        Tranform a DataSource timerange to the source-native timerange.

        args:
            time_range: TimeRange
                start and end datetime.
        """
        if dt := self.unit_add.get("time"):
            # subtract from source unit add
            return (time - timedelta(seconds=dt) for time in time_range)
        else:
            return time_range

    def to_source_variables(
        self, variables: Optional[List[str]]
    ) -> Optional[List[str]]:
        """
        Transform DataSource variables to the source-native names.

        args:
            variables: Optional[Variables]
                name(s) of the variables in the data.
        """
        if variables:
            inverse_rename_mapping: dict[str, str] = {
                v: k for k, v in self.rename.items()
            }
            return [inverse_rename_mapping.get(var, var) for var in variables]
