import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import json
import datetime
import pandas as pd


def mean_pooling(token_embeddings, attention_mask):
    """
    Effectively averages the embeddings of tokens across the vocabulary dimension
    to calculate the vocab-weighted latent representations (embeddings).
    :param token_embeddings: torch.float tensor of size (n_examples, n_vocab, n_latent)
    :param attention_mask: torch.byte tensor of size (n_examples, n_vocab)
    :return: torch.float tensor of size (n_examples, n_latent)
    """

    # return torch.mean(token_embeddings, dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def embed(query, tokenizer, model):
    """
    Embed `query` using `model` and return it.
    :param query: str of query
    :param tokenizer: HuggingFace tokenizer instance
    :param model: HuggingFace model instance
    """
    token = tokenizer([query], return_tensors='pt', truncation=True, padding=True)
    query_embedding = model(**token, output_hidden_states=True).hidden_states[-1]
    ## use pooling across vocab size
    query_embedding = mean_pooling(query_embedding, token['attention_mask'])
    return query_embedding


def cosine_similarity(v, M):
    """
    L2 similarity between a vector (single query embedding) and a matrix (of embeddings).
    :param v: torch.tensor of size (1, n_latent)
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: torch.tensor of size (n_documents,)
    """
    dv = v.norm(p=2)
    dM = M.norm(p=2, dim=1)
    return (M.matmul(v.T)).squeeze().div(dM * dv)


def return_ranked(query, tokenizer, model, M):
    """
    Embed a `query` using `model` and `tokenizer`, and return the
    indices of document embeddings `M` sorted most to least similar.
    :param query: str of query
    :param tokenizer: HuggingFace tokenizer instance
    :param model: HuggingFace model instance
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: a list of ints of length `n_documents`
    """
    q = embed(query, tokenizer, model)
    sims = cosine_similarity(q, M)
    rankings = torch.argsort(sims, descending=True)
    sims = sims[rankings].tolist()
    ranks = rankings.tolist()
    return list(zip(ranks, sims))


# Loads the vector embeddings
def load_embeddings(embeddings_path):
    m = torch.load(embeddings_path)
    return m


# Loads the tokenizer and masking model
def load_model(path_or_name):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForMaskedLM.from_pretrained(path_or_name)
    return tokenizer, model


def run_tool(query, min_words):
    # Loads json file containing UKRI grants
    f = open("data\\metadata.json")
    metadata = json.load(f)

    # Loads the vector embeddings
    embeddings = load_embeddings("data\\distilbert_ukri_tensor.pt")

    # Loads the tokenizer and masking model
    tokenizer, model = load_model("model\\distilbert_ukri")

    # Fetch results
    results = return_ranked(query, tokenizer, model, embeddings)

    # Subset to min words
    results = [r for r in results if len(metadata[str(r[0])]['abstract'].split()) > min_words]

    # Return data as json file
    meta = [{"reference": metadata[str(i)]["ref"],
             "funder": metadata[str(i)]["funder"],
             "title": metadata[str(i)]["title"],
             "abstract": metadata[str(i)]["abstract"],
             "value": int(metadata[str(i)]["funding"]),
             "n_words": len(metadata[str(i)]["abstract"].split()),
             "start_year": metadata[str(i)]["start"],
             "end_year": metadata[str(i)]["end"],
             "distance": dist}
            for i, dist in results[:-1]]

    # Turn json file into dataframe
    df = pd.DataFrame(meta)

    # Reference column is a list of grants. This will separate those grants out
    df = df.explode('reference')

    # time is unix. probably not a bad thing except all 00:00. Could fix this, but not essential,
    # so doing it here as not expensive.

    # some end dates are nan, so changing to 0
    df = df.fillna(0)

    # this could be fixed earlier, but the date columns are currently unix so changing to datetime
    df['start_year'] = pd.to_datetime(df['start_year'], unit='ms')
    df['start_year'] = pd.to_datetime(df['start_year']).dt.date

    df['end_year'] = pd.to_datetime(df['end_year'], unit='ms')
    df['end_year'] = pd.to_datetime(df['end_year']).dt.date

    return df
