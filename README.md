# Cohere Embed v3 on Amazon Bedrock — Mumbai Region

Working demonstrations of the Cohere Embed v3 models in **`ap-south-1`**.

Both models are invoked with their base model ID against the Mumbai Bedrock endpoint. No inference profile is involved, so inference runs in the India Region.

## Models

| Model ID | Use for | Dimensions |
|---|---|---|
| `cohere.embed-english-v3` | English content | 1024 |
| `cohere.embed-multilingual-v3` | Indic and mixed-language content | 1024 |

Embed v3 returns 1024-dimensional vectors. The dimension is fixed and cannot be configured. ([Cohere Embed v3 parameters][embed-v3], [AWS blog on embed-multilingual-v3][blog-multilingual])

Neither model supports a geographic or global inference profile, so the base model ID is the only invocation path. ([Embed English model card][card-english], [Embed Multilingual model card][card-multilingual])

## Demos

| File | Shows |
|---|---|
| `embed_english_v3.py` | Embedding English text |
| `embed_multilingual_v3.py` | Embedding Hindi, Tamil, and Bengali text |
| `semantic_search.py` | Search using `search_document` and `search_query` |

## Prerequisites

- Python 3.10+
- AWS credentials with `bedrock:InvokeModel` permission in `ap-south-1`

That is all an application role needs once the models are enabled in your account.

### One-time account enablement

Access to Bedrock foundation models is enabled by default, and Bedrock auto-enables a Marketplace-served model the first time it is invoked in an account. That auto-enablement step needs `aws-marketplace:Subscribe`, `aws-marketplace:Unsubscribe`, and `aws-marketplace:ViewSubscriptions`, plus a valid payment method on the account. ([Request access to models][model-access])

Per the AWS documentation, these permissions "are only required the first time a model is being used in an account." Once enabled, callers invoke the model with `bedrock:InvokeModel` alone and do not need Marketplace permissions. If your role cannot hold them, someone who can may enable the model once as an administrative step. ([Request access to models][model-access])

Check whether enablement is already done, in which case nothing further is needed:

```bash
aws bedrock get-foundation-model-availability \
  --model-id cohere.embed-multilingual-v3 \
  --region ap-south-1
```

`agreementAvailability.status` and `entitlementAvailability` of `AVAILABLE` mean the model is already enabled.

## Run

```bash
uv run --with boto3 python embed_english_v3.py
uv run --with boto3 python embed_multilingual_v3.py
uv run --with boto3 python semantic_search.py
```

## Verified output

All three demos were run against `ap-south-1` and returned 1024-dimensional embeddings.

`semantic_search.py` retrieved the correct document for each query:

```text
Query: How long does a refund take?
  0.644  Refunds are processed within five business days.

Query: When can I contact customer care?
  0.435  Our support team is available from 9 AM to 6 PM IST.

Query: How do I change where my order is delivered?
  0.518  Use the app settings page to update your address.
```

## Understanding `input_type`

`input_type` is a required parameter for Embed v3. Per the AWS documentation, it "prepends special tokens to differentiate each type from one another," so the same text produces a differently optimized vector depending on the value. ([Cohere Embed v3 parameters][embed-v3])

| Value | Use for |
|---|---|
| `search_document` | Text you store in your vector database |
| `search_query` | The user's search query at lookup time |
| `classification` | Input to a text classifier |
| `clustering` | Grouping similar texts together |

The retrieval pair matters most. AWS guidance is to "embed your corpus with the `search_document` type and embedded queries with type `search_query` type." This supports asymmetric search, where a short question and a longer answer passage look very different as raw text but still need to land close together in vector space. ([Cohere Embed v3 parameters][embed-v3])

This is why `semantic_search.py` calls the API twice:

```python
document_vectors = embed(DOCUMENTS, "search_document")  # indexing
query_vectors    = embed(QUERIES,   "search_query")     # lookup
```

Two practical consequences:

- Using the same value on both sides reduces retrieval quality, because it discards the asymmetry the model was trained to exploit.
- The value chosen at indexing time is baked into the stored vectors. Changing it later requires re-embedding the whole corpus, so decide up front.

The documentation also advises that you "should not mix different types together," except for the search and retrieval pairing above. ([Cohere Embed v3 parameters][embed-v3])

## Usage notes

All limits below are from the Cohere Embed v3 parameters reference. ([Cohere Embed v3 parameters][embed-v3])

- Maximum 96 texts per request. Split larger sets into multiple calls.
- Each text is limited to 512 tokens, roughly 2,048 characters.
- `truncate` accepts `NONE`, `START`, or `END`. Add `"truncate": "END"` to trim longer inputs instead of receiving an error. Note these values differ from Embed v4, which uses `LEFT` and `RIGHT`.
- `embedding_types` accepts `float`, `int8`, `uint8`, `binary`, and `ubinary`. Using `int8` reduces vector size and storage cost.
- Amazon Bedrock does not support streaming responses from Cohere Embed models. ([Cohere Embed models overview][embed-overview])

## References

- [Cohere Embed v3 inference parameters][embed-v3] — request and response schema, `input_type` values, text and token limits
- [Cohere Embed and Cohere Embed v4 models][embed-overview] — invocation overview and streaming limitation
- [Embed English model card][card-english] — model ID and supported endpoints
- [Embed Multilingual model card][card-multilingual] — model ID and supported endpoints
- [Inference using Invoke API][inference-api] — `modelId` selection for base models and inference profiles
- [Request access to models][model-access] — AWS Marketplace prerequisites for model access
- [boto3 `invoke_model` reference][boto3-invoke] — Python API signature
- [Build financial search applications using the Amazon Bedrock Cohere multilingual embedding model][blog-multilingual] — confirms 1,024 dimensions for `embed-multilingual-v3`

[embed-v3]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
[embed-overview]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
[card-english]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-english.html
[card-multilingual]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-multilingual.html
[inference-api]: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-api.html
[model-access]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html
[boto3-invoke]: https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock-runtime/client/invoke_model.html
[blog-multilingual]: https://aws.amazon.com/blogs/machine-learning/build-financial-search-applications-using-the-amazon-bedrock-cohere-multilingual-embedding-model/

## Disclaimer

The sample code in this repository is provided for demonstration purposes only. It is not intended for production use and is provided "as is" without warranty of any kind. You are responsible for reviewing the code, testing it in your own environment, and any charges incurred by running it. Verify current model behaviour, limits, and Regional availability against the AWS documentation linked above.
