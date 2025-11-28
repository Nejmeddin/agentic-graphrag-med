from neo4j import GraphDatabase
from med_graphrag.config.settings import settings

def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
