# TODO: use pydantic for other modules in Qlib
from pydantic import BaseSettings


class Conf(BaseSettings):
    """module specific settings."""

    ...
