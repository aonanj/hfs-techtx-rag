from typing import List, Tuple
import json

from infrastructure.embeddings import get_chroma_collection
from infrastructure.logger import get_logger

logger = get_logger()

def find_nearest_neighbors(
    query: str,
    num_neighbors: int,
) -> List[Tuple[str, float]]:
    """
    Finds the nearest neighbors for a given query using ChromaDB.

    Args:
        query: The query string.
        num_neighbors: The number of nearest neighbors to retrieve.

    Returns:
        A list of tuples, where each tuple contains the ID of a neighbor and its distance.
    """
    logger.info(f"Finding {num_neighbors} nearest neighbors for query: '{query}'")
    
    collection = get_chroma_collection()
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=num_neighbors
        )
        
        if (
            not results
            or not results.get("ids")
            or not results["ids"][0]
            or not results.get("distances")
            or not results["distances"]
        ):
            logger.error("No neighbors found or results format is invalid.")
            return []

        ids = results["ids"][0]
        distances = results["distances"][0]
        
        neighbors = list(zip(ids, distances))
        logger.info(f"Found {len(neighbors)} neighbors.")
        return neighbors

    except Exception as e:
        logger.error(f"ChromaDB neighbor search failed: {e}")
        return []


def upsert_datapoints(datapoints: list[dict]):
    """Upserts datapoints to the ChromaDB collection."""
    if not datapoints:
        logger.info("No datapoints to upsert.")
        return

    collection = get_chroma_collection()
    
    ids = [str(dp['datapoint_id']) for dp in datapoints]
    embeddings = [dp['embedding'] for dp in datapoints]
    
    # ChromaDB metadatas must be Dict[str, str | int | float | bool]
    # Our 'restricts' are lists of dicts, which are not supported.
    # We will serialize them to JSON strings.
    metadatas = []
    for dp in datapoints:
        metadata = {}
        if 'restricts' in dp:
            # Assuming 'restricts' is a list of dictionaries
            metadata['restricts'] = json.dumps(dp['restricts'])
        if 'numeric_restricts' in dp:
            metadata['numeric_restricts'] = json.dumps(dp['numeric_restricts'])
        metadatas.append(metadata)

    try:
        logger.info(f"Upserting {len(datapoints)} datapoints to ChromaDB collection.")
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Successfully upserted {len(datapoints)} datapoints.")
    except Exception as e:
        logger.error(f"Failed to upsert to ChromaDB: {e}")
        raise Exception("Failed to upsert to ChromaDB")
