"""Semantic search with Cohere Embed v3 in the Mumbai Region (ap-south-1).

Shows the correct pairing of input types:
  - Documents are embedded with input_type="search_document"
  - Queries are embedded with input_type="search_query"

AWS guidance: "embed your corpus with the search_document type and embedded
queries with type search_query type."
https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html

Run:
    uv run --with boto3 python semantic_search.py
"""

import json
import math

import boto3

REGION = "ap-south-1"
MODEL_ID = "cohere.embed-multilingual-v3"

DOCUMENTS = [
    "Refunds are processed within five business days.",
    "Our support team is available from 9 AM to 6 PM IST.",
    "You can track your order status in the mobile app.",
    "Free delivery applies to orders above 500 rupees.",
    "Use the app settings page to update your address.",
]

QUERIES = [
    "How long does a refund take?",
    "When can I contact customer care?",
    "How do I change where my order is delivered?",
]

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def embed(texts, input_type):
    """Embed texts in ap-south-1 and return the float vectors."""
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        body=json.dumps(
            {
                "texts": texts,
                "input_type": input_type,
                "embedding_types": ["float"],
            }
        ),
    )
    return json.loads(response["body"].read())["embeddings"]["float"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    return dot / (magnitude_a * magnitude_b)


def main():
    document_vectors = embed(DOCUMENTS, "search_document")
    query_vectors = embed(QUERIES, "search_query")

    print(f"Region     : {REGION}")
    print(f"Model      : {MODEL_ID}")
    print(f"Dimensions : {len(document_vectors[0])}")
    print(f"Documents  : {len(DOCUMENTS)}")

    for query, query_vector in zip(QUERIES, query_vectors):
        scores = [
            (cosine_similarity(query_vector, document_vector), document)
            for document_vector, document in zip(document_vectors, DOCUMENTS)
        ]
        scores.sort(reverse=True)

        print(f"\nQuery: {query}")
        for score, document in scores[:2]:
            print(f"  {score:.3f}  {document}")


if __name__ == "__main__":
    main()
