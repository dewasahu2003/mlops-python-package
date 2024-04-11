"""Define and validate dataframe schemas."""

# %% IMPORTS

import typing as T

import pandas as pd
import pandera as pa
import pandera.typing as papd
import pandera.typing.common as padt
from pydantic import constr


# %% TYPES

# Generic type for a dataframe container
TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")

# %% SCHEMAS


class Schema(pa.DataFrameModel):
    """Base class for a dataframe schema.

    Use a schema to type your dataframe object.
    e.g., to communicate and validate its fields.
    """

    class Config:
        """Default configurations for all schemas.

        Parameters:
            coerce (bool): convert data type if possible.
            strict (bool): ensure the data type is correct.
        """

        coerce: bool = True
        strict: bool = True

    @classmethod
    def check(cls: T.Type[TSchema], data: pd.DataFrame) -> papd.DataFrame[TSchema]:
        """Check the dataframe with this schema.

        Args:
            data (pd.DataFrame): dataframe to check.

        Returns:
            papd.DataFrame[TSchema]: validated dataframe.
        """
        return T.cast(papd.DataFrame[TSchema], cls.validate(data))


class InputsSchema(Schema):
    """Schema for the project inputs."""

    CustomerID: papd.Index[padt.UInt32] = pa.Field(ge=0)
    Genre: papd.Series[padt.UInt32] = pa.Field()
    Age: papd.Series[padt.UInt32] = pa.Field()
    Annual_Income: papd.Series[padt.Float16] = pa.Field(ge=0)


Inputs = papd.DataFrame[InputsSchema]


class TargetsSchema(Schema):
    """Schema for the project target."""

    CustomerID: papd.Index[padt.UInt32] = pa.Field(ge=0)
    Spending_Score: papd.Series[padt.UInt32] = pa.Field(ge=0, le=100)


Targets = papd.DataFrame[TargetsSchema]


class OutputsSchema(Schema):
    """Schema for the project output."""

    CustomerID: papd.Index[padt.UInt32] = pa.Field(ge=0)
    prediction: papd.Series[padt.UInt32] = pa.Field(ge=0)


Outputs = papd.DataFrame[OutputsSchema]
