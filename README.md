# Sequence Models with PyTorch

Implementations and notebooks for learning recurrent neural networks, language modelling, and English-to-French machine translation.

## Contents

| Notebook | Topic |
| --- | --- |
| [Sequences](./notebooks/sequences.ipynb) | Sequence fundamentals |
| [Raw to sequence data](./notebooks/raw_to_sequence_data.ipynb) | Loading, tokenising, and batching text |
| [Language model](./notebooks/language_model.ipynb) | Character-level language modelling |
| [Machine translation](./notebooks/machine_translation.ipynb) | GRU encoder-decoder with teacher forcing |

Model implementations are under [`src/`](./src).

## Run the notebooks

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab
```

## Results

### English-to-French translation

GRU encoder-decoder validation BLEU-4: **32.80**.

```text
Input:
The house is on the hill.
There is no one there.
I think we should go back.

Output:
la maison est sur la colline.
il n'y a personne.
je pense que nous devrions y retourner.
```

The notebook uses a custom add-one-smoothed BLEU implementation, so this score should only be compared with runs using the same evaluator.

### Character-level language model

```python
text, states = generate_char_stream(model, vocab, "this is a pro", stream_length=10)
```

```text
this is a problem to me
```
