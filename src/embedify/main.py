#!/usr/bin/env python3
from pathlib import Path
from typing import Annotated, Optional
import typer
import tomllib
from embedify.config import get_migration_client, load_config
from embedify.types import InstalledVectorDb

app = typer.Typer()


@app.command()
def embed():
    pass


@app.command()
def installed_vector_dbs():
    typer.echo("Installed Vector Databases:")
    for db in InstalledVectorDb:
        typer.echo(f" {db.value}")


@app.command()
def migrate(config_file: Annotated[Path, typer.Argument], limit: Optional[int] = 10):
    if not config_file.is_file():
        raise ValueError(f"Config file not found: {config_file}")
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)
    config = load_config(config_data)
    client = get_migration_client(config)
    client.migrate(limit=limit)


if __name__ == "__main__":
    app()
