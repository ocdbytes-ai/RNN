# Sequences

This repository focuses upon sequences in RNNs and Language Model.

## Notebooks

- [Sequences Introduction and Concepts](./notebooks/sequences.ipynb)
- [Raw to Sequenced Data](./notebooks/raw_to_sequence_data.ipynb)
- [Language Model](./notebooks/language_model.ipynb)

## Result 

```py
text, states = generate_char_stream(model, vocab, "this is a pro", stream_length=10)
```

Result :

```
this is a problem to me
```