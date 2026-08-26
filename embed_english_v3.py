"""Cohere Embed English v3 on Amazon Bedrock in the Mumbai Region (ap-south-1).

Returns 1024-dimensional embeddings. The dimension is fixed for Embed v3.

References:
  Cohere Embed v3 inference parameters
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html

  Embed English model card
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-english.html

Run:
    uv run --with boto3 python embed_english_v3.py
"""

import json

import boto3

REGION = "ap-south-1"
MODEL_ID = "cohere.embed-english-v3"

TEXTS = [
    "Our support team is available from 9 AM to 6 PM IST.",
    "You can track your order in the mobile app.",
    "Refunds are processed within five business days.",
]


def main():
    # The client is created for ap-south-1, so the request goes to the Mumbai endpoint.
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        body=json.dumps(
            {
                "texts": TEXTS,
                "input_type": "search_document",
                "embedding_types": ["float"],
            }
        ),
    )

    result = json.loads(response["body"].read())
    embeddings = result["embeddings"]["float"]

    print(f"Region     : {REGION}")
    print(f"Model      : {MODEL_ID}")
    print(f"Texts      : {len(TEXTS)}")
    print(f"Embeddings : {len(embeddings)}")
    print(f"Dimensions : {len(embeddings[0])}")
    print()

    for text, vector in zip(TEXTS, embeddings):
        preview = ", ".join(f"{value:.4f}" for value in vector[:4])
        print(f"  {text}")
        print(f"    -> [{preview}, ...] ({len(vector)} dimensions)")


if __name__ == "__main__":
    main()
