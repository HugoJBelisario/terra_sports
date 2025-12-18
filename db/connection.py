import os
import psycopg2


def get_connection():
    """
    Returns a PostgreSQL connection using environment variables.
    """
    required_vars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [v for v in required_vars if not os.getenv(v)]

    if missing:
        raise EnvironmentError(
            f"Missing required database environment variables: {', '.join(missing)}"
        )

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", 5432)),
        sslmode="require",
    )