"""CLI integration for DeepData with VectorDBBench."""

from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class DeepDataTypedDict(CommonTypedDict):
    url: Annotated[
        str,
        click.option(
            "--url",
            type=str,
            help="DeepData server URL",
            default="http://localhost:8080",
        ),
    ]
    api_key: Annotated[
        str | None,
        click.option(
            "--api-key",
            type=str,
            help="DeepData API key",
            required=False,
        ),
    ]
    m: Annotated[
        int,
        click.option(
            "--m",
            type=int,
            help="HNSW M parameter (max connections)",
            default=16,
        ),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            help="HNSW ef_construction parameter",
            default=200,
        ),
    ]
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=int,
            help="HNSW ef_search parameter",
            default=128,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(DeepDataTypedDict)
def DeepData(**parameters: Unpack[DeepDataTypedDict]):
    """Run VectorDBBench benchmarks against DeepData."""
    from .config import DeepDataConfig, DeepDataIndexConfig

    config_params = {
        "db_label": parameters.get("db_label", ""),
        "url": parameters["url"],
    }
    if parameters.get("api_key"):
        config_params["api_key"] = SecretStr(parameters["api_key"])

    run(
        db=DB.DeepData,
        db_config=DeepDataConfig(**config_params),
        db_case_config=DeepDataIndexConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
