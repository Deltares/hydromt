from pathlib import Path
from typing import List, Type

import pandas as pd
import pytest
from pydantic import ValidationError

from hydromt._typing import StrPath
from hydromt.data_adapter import DataFrameAdapter
from hydromt.data_source import DataFrameSource
from hydromt.drivers import DataFrameDriver
from hydromt.metadata import SourceMetadata


@pytest.fixture(scope="session")
def mock_df_adapter():
    class MockDataFrameAdapter(DataFrameAdapter):
        def transform(self, df: pd.DataFrame, metadata: SourceMetadata, **kwargs):
            return df

    return MockDataFrameAdapter()


@pytest.fixture(scope="class")
def MockDataFrameDriver(df: pd.DataFrame) -> Type[DataFrameDriver]:
    class MockDataFrameDriver(DataFrameDriver):
        name = "mock_df_driver"

        def write(self, path: StrPath, df: pd.DataFrame, **kwargs) -> None:
            pass

        def read(self, uri: str, **kwargs) -> pd.DataFrame:
            return self.read_data([uri], **kwargs)

        def read_data(self, uris: List[str], **kwargs) -> pd.DataFrame:
            return df

    return MockDataFrameDriver


class TestDataFrameSource:
    def test_validators(self, mock_df_adapter: DataFrameAdapter):
        with pytest.raises(ValidationError) as e_info:
            DataFrameSource(
                name="name",
                uri="uri",
                data_adapter=mock_df_adapter,
                driver="does not exist",
            )

        assert e_info.value.error_count() == 1
        error_driver = next(
            filter(lambda e: e["loc"] == ("driver",), e_info.value.errors())
        )
        assert error_driver["type"] == "model_type"

    def test_model_validate(
        self,
        MockDataFrameDriver: Type[DataFrameDriver],
        mock_df_adapter: DataFrameAdapter,
    ):
        DataFrameSource.model_validate(
            {
                "name": "example_file",
                "driver": MockDataFrameDriver(),
                "data_adapter": mock_df_adapter,
                "uri": "test_uri",
            }
        )
        with pytest.raises(ValidationError, match="'data_type' must be 'DataFrame'."):
            DataFrameSource.model_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "DifferentDataType",
                    "driver": MockDataFrameDriver(),
                    "data_adapter": mock_df_adapter,
                    "uri": "test_uri",
                }
            )

    def test_instantiate_directly(
        self,
    ):
        datasource = DataFrameSource(
            name="test",
            uri="points.csv",
            driver={"name": "pandas", "metadata_resolver": "convention"},
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )
        assert isinstance(datasource, DataFrameSource)

    def test_instantiate_directly_minimal_kwargs(self):
        DataFrameSource(
            name="test",
            uri="points.csv",
            driver={"name": "pandas"},
        )

    def test_read_data(
        self,
        MockDataFrameDriver: Type[DataFrameDriver],
        mock_df_adapter: DataFrameAdapter,
        df: pd.DataFrame,
        tmp_dir: Path,
    ):
        tmp_dir.touch("test.xls")
        source = DataFrameSource(
            root=".",
            name="example_source",
            driver=MockDataFrameDriver(),
            data_adapter=mock_df_adapter,
            uri=str(tmp_dir / "test.xls"),
        )
        pd.testing.assert_frame_equal(df, source.read_data())

    def test_to_file(self, MockDataFrameDriver: Type[DataFrameDriver]):
        mock_df_driver = MockDataFrameDriver()
        source = DataFrameSource(
            name="test",
            uri="source.csv",
            driver=mock_df_driver,
            metadata=SourceMetadata(crs=4326),
        )
        new_source = source.to_file("test")
        assert "local" in new_source.driver.filesystem.protocol
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(mock_df_driver) != id(new_source.driver)

    def test_to_file_override(self, MockDataFrameDriver: Type[DataFrameDriver]):
        driver1 = MockDataFrameDriver()
        source = DataFrameSource(
            name="test",
            uri="df.xlsx",
            driver=driver1,
            metadata=SourceMetadata(category="test"),
        )
        driver2 = MockDataFrameDriver(filesystem="memory")
        new_source = source.to_file("test", driver_override=driver2)
        assert new_source.driver.filesystem.protocol == "memory"
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(driver2) == id(new_source.driver)
