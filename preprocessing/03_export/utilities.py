import numpy as np
import pandas as pd
import psycopg2
import sqlalchemy
from joblib import Memory 
from psycopg2.extensions import register_adapter, AsIs

import config

cache = Memory(".cache", verbose=1)

### type adapters for python types --> postgres
def adapt_numpy_float64(numpy_float64):
    if np.isnan(numpy_float64):
        return psycopg2.extensions.AsIs('NULL')
    return AsIs(numpy_float64)
def adapt_numpy_int64(numpy_int64):
    if np.isnan(numpy_int64):
        return psycopg2.extensions.AsIs('NULL')
    return AsIs(numpy_int64)
def adapt_numpy_bool(numpy_bool):
    if np.isnan(numpy_bool):
        return psycopg2.extensions.AsIs('NULL')
    if numpy_bool:
        return psycopg2.extensions.AsIs('True')
    return psycopg2.extensions.AsIs('False')

def adapt_set(s):
    if not len(s):
        return psycopg2.extensions.AsIs("'{}'")
    return psycopg2.extensions.QuotedString(str(s))
    
register_adapter(np.float64, adapt_numpy_float64)
register_adapter(set, adapt_set)
register_adapter(np.bool_, adapt_numpy_bool)
register_adapter(np.int64, adapt_numpy_int64)

def get_connection() -> sqlalchemy.Engine:
    """Creates a SQLAlchemy engine to access the database.

    Returns
    -------
    sqlalchemy.Engine
    """
    return sqlalchemy.create_engine(
        "postgresql://{username}:{password}@{url}:{port}/{db_name}".format(
            username=config.username,
            url=config.url,
            password=config.password,
            db_name=config.db_name,
            port=config.port,
        )
    )


def get_query(name: str) -> str:
    """Load the text of a SQL query saved in the sql/ folder.

    Parameters
    ----------
    name : str
        Name of the query without the sql/ part or the file extension.

    Returns
    -------
    str
        SQL query.
    """
    with open(f"../sql/{name}.sql", "r") as f:
        return f.read()


def run_sql(filename, cache_result=True, **kwargs):
    """Run an SQL file located in project_dir/sql
    passes on **kwargs to the underlying pd.read_sql
    Per default, results will be cached. 
    Avoid cache by using cache_result=False"""
    query = get_query(filename)
    if cache_result:
        df = run_sql_query(query, **kwargs)
    else: 
        df = run_sql_query_uncached(query, **kwargs)
    return df


@cache.cache
def run_sql_query(query, **kwargs):
    conn = get_connection()
    df = pd.read_sql(query, conn, **kwargs)
    return df


def run_sql_query_uncached(query, **kwargs):
    conn = get_connection()
    df = pd.read_sql(query, conn, **kwargs)
    return df

