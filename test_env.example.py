import os
from dotenv import load_dotenv
print("CWD =", os.getcwd())
print("db_config.env exists?", os.path.exists("db_config.env"))
load_dotenv("db_config.env")
print("PGUSER =", os.getenv("PGUSER"))
print("PGHOST =", os.getenv("PGHOST"))
print("PGPORT =", os.getenv("PGPORT"))
print("PGDB   =", os.getenv("PGDB"))


