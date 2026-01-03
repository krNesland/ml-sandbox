"""
Implemented from: https://ngrok.com/blog/prompt-caching/

Demonstrating the attention mechanism with a toy example.

Skipping the positional encoding (and probably a lot more).

Also, adding a rough sketch of the latter stages of the LLM forward pass.
"""

import numpy as np
import torch

torch.manual_seed(1337)


def tensor_to_str(tensor, decimals=3):
    arr = tensor.detach().cpu().numpy()

    # Use numpy's array2string for nice formatting
    return np.array2string(
        arr, precision=decimals, suppress_small=True, separator=", ", floatmode="fixed"
    )


def multi_headed_attention(embeddings):
    wq = torch.randn(dim, dim)  # Query matrix. Learnt during training.
    wk = torch.randn(dim, dim)  # Key matrix. Learnt during training.
    wv = torch.randn(dim, dim)  # Value matrix. Learnt during training.

    q = embeddings @ wq
    assert q.shape == (num, dim)
    print("\nq\n", tensor_to_str(q))

    k = embeddings @ wk
    assert k.shape == (num, dim)
    print("\nk\n", tensor_to_str(k))

    scores = q @ k.T
    assert scores.shape == (num, num)
    print("\nscores\n", tensor_to_str(scores))

    mask = torch.tril(torch.ones(num, num)).bool()
    assert mask.shape == (num, num)
    print("\nmask\n", tensor_to_str(mask))

    # Masking the scores to only allow attention to previous tokens. Do not want future tokens to influence past tokens
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    assert masked_scores.shape == (num, num)
    print("\nmasked_scores\n", tensor_to_str(masked_scores))

    weights = torch.softmax(masked_scores, dim=-1)
    print("\nweights\n", tensor_to_str(weights))

    v = embeddings @ wv
    assert v.shape == (num, dim)
    print("\nv\n", tensor_to_str(v))

    attention_output = weights @ v
    assert attention_output.shape == (num, dim)
    print("\nattention_output\n", tensor_to_str(attention_output))

    return attention_output


if __name__ == "__main__":
    dim = 3  # Embedding dimensions
    num = 4  # Number of tokens in the input sequence
    vocab_tokens = [  # All possible tokens in the vocabulary
        "hello ",
        "kristoffer ",
        "puma ",
        "green ",
        "blue ",
        "moon ",
        "racing ",
        "? ",
    ]
    print("\nvocab_tokens\n", vocab_tokens)

    vocab_size = len(vocab_tokens)

    vocab_embeddings = torch.randn(
        vocab_size, dim
    )  # Embeddings for all tokens in the vocabulary
    print("\nvocab_embeddings\n", tensor_to_str(vocab_embeddings))

    selected_token_indices = torch.randint(
        0, vocab_size, (num,)
    )  # Selecting random tokens from the vocabulary
    print("\nselected_token_indices\n", selected_token_indices)

    input_sequence = "".join([vocab_tokens[i] for i in selected_token_indices])
    print("\ninput_sequence\n", input_sequence)

    selected_embeddings = vocab_embeddings[
        selected_token_indices
    ]  # Converting the randomly selected tokens to embeddings
    print("\nselected_embeddings\n", tensor_to_str(selected_embeddings))

    attention_output = multi_headed_attention(selected_embeddings)

    # Mainly interested in the multi headed attention, but adding (below) a rough sketch of the latter stages of the LLM forward pass.
    # However, skipping some residual connections and layer normalization for brevity.

    ffn_dim = dim * 2
    w1 = torch.randn(dim, ffn_dim)
    w2 = torch.randn(ffn_dim, dim)
    ffn_output = torch.nn.functional.relu(attention_output @ w1) @ w2
    print("\nffn_output\n", tensor_to_str(ffn_output))

    tranformer_block_output = ffn_output

    output_projection = torch.randn(dim, vocab_size)

    logits = tranformer_block_output @ output_projection
    print("\nlogits\n", tensor_to_str(logits))

    temperature = 2.0  # Temperature > 1.0 makes the model more random, < 1.0 makes it more deterministic.

    probs = torch.softmax(logits / temperature, dim=-1)
    print("\nprobs\n", tensor_to_str(probs))

    next_token_index = torch.multinomial(probs[-1], num_samples=1)
    print("\nnext_token_index\n", tensor_to_str(next_token_index))

    next_token = vocab_tokens[next_token_index]
    print("\nnext_token\n", next_token)
