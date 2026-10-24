# sparknlp-1455-helpers

Helper files for the SPARKNLP-1455 benchmark notebook, which compares Polish → German translation in Spark NLP
(M2M100, Marian, NLLB) with Hugging Face Transformers on a Colab CPU runtime. The notebook downloads them with
`wget` in its setup section.

- `bench_utils.py`: test data (FLORES-200), the matched decoding settings, starting and stopping Spark, the Spark NLP
  and Hugging Face runs, CPU and memory sampling, scoring, and saving results.
- `bench_charts.py`: the summary table and charts.
