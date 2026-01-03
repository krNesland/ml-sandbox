"""
Educational Transformer Forward Pass Implementation

A minimal implementation of a transformer's forward pass, demonstrating:
1. Token embedding lookup
2. Self-attention mechanism (simplified single-head)
3. Feed-forward network
4. Token prediction with temperature sampling

Implemented from: https://ngrok.com/blog/prompt-caching/

Note: Skips positional encoding, residual connections, and layer normalization for clarity.
"""

import numpy as np
import torch

torch.manual_seed(1337)


def tensor_to_str(tensor, decimals=3):
    """Format tensors for readable console output."""
    arr = tensor.detach().cpu().numpy()

    # Use numpy's array2string for nice formatting
    return np.array2string(
        arr, precision=decimals, suppress_small=True, separator=", ", floatmode="fixed"
    )


def self_attention(embeddings):
    """
    Breaking this up into a separate function because this is the key part of the LLM.

    Compute self-attention: allows each token to attend to previous tokens.

    Steps:
    1. Project embeddings to Query, Key, Value spaces
    2. Compute attention scores (Q @ K^T)
    3. Apply causal mask (prevent attending to future tokens)
    4. Softmax to get attention weights
    5. Weight the Values to get attention output

    Note: This is single-head attention. Multi-head would compute this
    multiple times in parallel and concatenate the results.
    """
    wq = torch.randn(dim, dim)  # Query matrix. Learnt during training.
    wk = torch.randn(dim, dim)  # Key matrix. Learnt during training.
    wv = torch.randn(dim, dim)  # Value matrix. Learnt during training.

    # Query: "what am I looking for?"
    q = embeddings @ wq
    assert q.shape == (num, dim)
    print("\nq\n", tensor_to_str(q))

    # Key: "what do I contain?"
    k = embeddings @ wk
    assert k.shape == (num, dim)
    print("\nk\n", tensor_to_str(k))

    # Attention scores: how much should each token attend to others?
    scores = q @ k.T
    assert scores.shape == (num, num)
    print("\nscores\n", tensor_to_str(scores))

    # Causal mask: only attend to previous tokens (lower triangular)
    mask = torch.tril(torch.ones(num, num)).bool()
    assert mask.shape == (num, num)
    print("\nmask\n", tensor_to_str(mask))

    # Apply mask: set future positions to -inf so they become 0 after softmax
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    assert masked_scores.shape == (num, num)
    print("\nmasked_scores\n", tensor_to_str(masked_scores))

    # Normalize scores to get attention weights (probabilities)
    weights = torch.softmax(masked_scores, dim=-1)
    print("\nweights\n", tensor_to_str(weights))

    # Value: "what do I communicate?"
    v = embeddings @ wv
    assert v.shape == (num, dim)
    print("\nv\n", tensor_to_str(v))

    # Weighted sum of values based on attention weights
    attention_output = weights @ v
    assert attention_output.shape == (num, dim)
    print("\nattention_output\n", tensor_to_str(attention_output))

    return attention_output


if __name__ == "__main__":
    # ==== Setup: Define vocabulary and embeddings ====
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

    # Each token gets mapped to a learned embedding vector
    vocab_embeddings = torch.randn(
        vocab_size, dim
    )  # Embeddings for all tokens in the vocabulary
    print("\nvocab_embeddings\n", tensor_to_str(vocab_embeddings))

    # ==== Step 1: Tokenize and embed input ====
    selected_token_indices = torch.randint(
        0, vocab_size, (num,)
    )  # Selecting random tokens from the vocabulary
    print("\nselected_token_indices\n", selected_token_indices)

    input_sequence = "".join([vocab_tokens[i] for i in selected_token_indices])
    print("\ninput_sequence\n", input_sequence)

    # Look up embeddings for our input tokens
    selected_embeddings = vocab_embeddings[
        selected_token_indices
    ]  # Converting the randomly selected tokens to embeddings
    print("\nselected_embeddings\n", tensor_to_str(selected_embeddings))

    # ==== Step 2: Self-attention mechanism ====
    attention_output = self_attention(selected_embeddings)

    # ==== Step 3: Feed-forward network ====
    # Simple 2-layer FFN: expand, ReLU, contract
    ffn_dim = dim * 2
    w1 = torch.randn(dim, ffn_dim)  # Learnt during training
    w2 = torch.randn(ffn_dim, dim)  # Learnt during training
    ffn_output = torch.nn.functional.relu(attention_output @ w1) @ w2
    print("\nffn_output\n", tensor_to_str(ffn_output))

    tranformer_block_output = ffn_output  # Note: skipping residuals & layer norm

    # ==== Step 4: Project to vocabulary and sample next token ====
    output_projection = torch.randn(dim, vocab_size)  # Learnt during training

    # Logits: unnormalized scores for each possible next token
    logits = tranformer_block_output @ output_projection
    print("\nlogits\n", tensor_to_str(logits))

    # Temperature controls randomness: >1 more random, <1 more deterministic
    temperature = 2.0

    # Convert logits to probabilities
    probs = torch.softmax(logits / temperature, dim=-1)
    print("\nprobs\n", tensor_to_str(probs))

    # Sample the next token from the probability distribution
    next_token_index = torch.multinomial(probs[-1], num_samples=1)
    print("\nnext_token_index\n", tensor_to_str(next_token_index))

    next_token = vocab_tokens[next_token_index]
    print("\nnext_token\n", next_token)
