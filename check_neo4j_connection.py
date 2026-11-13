from getpass import getpass
from neo4j import GraphDatabase

uri = input("Aura URI: ").strip()
username = input("Aura username: ").strip()
password = getpass("Aura password: ")
database = input("Database name [neo4j]: ").strip() or "neo4j"

with GraphDatabase.driver(uri, auth=(username, password)) as driver:
    driver.verify_connectivity()
    with driver.session(database=database) as session:
        result = session.run("RETURN 1 AS ok").single()
        print("Connected. Read-only check:", result["ok"])