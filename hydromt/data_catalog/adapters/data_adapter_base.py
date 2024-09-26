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

    def _to_source_timerange(
        self,
        time_range: Optional[TimeRange],
    ) -> Optional[TimeRange]:
        """Transform a DataSource timerange to the source-native timerange.

        Parameters
        ----------
        time_range : Optional[TimeRange]
            start and end datetime

        Returns
        -------
        Optional[TimeRange]
            time_range in source format
        """
        if time_range is None:
            return None
        elif dt := self.unit_add.get("time"):
            # subtract from source unit add
            return (time - timedelta(seconds=dt) for time in time_range)
        else:
            return time_range

    def _to_source_variables(
        self, variables: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Transform DataSource variables to the source-native names.

        Parameters
        ----------
        variables : Optional[List[str]]
            name(s) of the variables in the data.

        Returns
        -------
        Optional[List[str]]
            _description_
        """
        if variables:
            inverse_rename_mapping: dict[str, str] = {
                v: k for k, v in self.rename.items()
            }
            return [inverse_rename_mapping.get(var, var) for var in variables]
