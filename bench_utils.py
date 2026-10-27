"""Helpers for the SPARKNLP-1455 notebook. Downloaded to /content by the setup cell; every section imports it.

Everything a section needs after a session restart comes from files under ROOT, never from notebook variables.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import json
import os
import pathlib
import threading
import time
import urllib.request

ROOT = os.environ.get("SN1455_ROOT", "/content")
RESULTS_DIR = os.path.join(ROOT, "sparknlp_1455_results")
MODELS_DIR = os.path.join(ROOT, "models")
EXPORT_DIR = os.path.join(ROOT, "export")
DATA_TSV = os.path.join(ROOT, "pl_de_flores.tsv")
CONFIG_JSON = os.path.join(ROOT, "sn1455_config.json")
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

N_FULL, N_SMOKE, N_SINGLE = 200, 20, 20
MAX_OUT = 256
SPARK_MEMORY = "8G"  # driver heap cap; Spark NLP's 16G default would compete with the HF model on Colab's ~12.7 GB
ORDER = ("m2m100", "marian", "nllb")

MODELS = {
    "m2m100": dict(hf_id="facebook/m2m100_418M", precision="int8",
                   onnx=("Xenova/m2m100_418M", "9c374f0b7aca709787cea97b047bfbbd1559d177"),
                   annotator="M2M100Transformer", src="pl", tgt="de", max_input=1024, batch=1),
    "marian": dict(hf_id="Helsinki-NLP/opus-mt-pl-de", precision="fp32", task="text2text-generation-with-past",
                   annotator="MarianTransformer", src=None, tgt=None, max_input=512, batch=4),
    "nllb": dict(hf_id="facebook/nllb-200-distilled-600M", precision="int8",
                 onnx=("Xenova/nllb-200-distilled-600M", "261c31d1a5732c67cdd16d80e8d6088507c7ccea"),
                 annotator="NLLBTransformer", src="pol_Latn", tgt="deu_Latn", max_input=1024, batch=1),
}
# Batch sizes, the same on both sides: Spark NLP 7.0.0's M2M100/NLLB can't batch (inputs aren't padded, and only the
# first row of each batch gets a result), so they run one sentence at a time; Marian pads and batches correctly.
# Precision, the same on both sides. M2M100 and NLLB run at int8: Spark NLP loads ready-made int8 ONNX builds of the
# same checkpoints ("onnx": repo and pinned commit; ONNX Runtime dynamic quantization, int8 weights per channel), and
# Hugging Face quantizes the checkpoint with PyTorch's dynamic int8. Exporting NLLB at fp32 needs about 14 GB, more
# than Colab's 12.7 GB, and its fp32 decoder is over ONNX's 2 GB single-file limit, which Spark NLP 7.0.0 can't load.
# Marian is small and is exported at fp32 ("task": optimum's export task).
INT8_FILES = {"encoder_model.onnx": "onnx/encoder_model_quantized.onnx",
              "decoder_model.onnx": "onnx/decoder_model_quantized.onnx"}

TICKET_PL = "Cześć, jak się masz? Mam nadzieję, że masz wspaniały dzień."
QUALITATIVE = [
    ("ticket", TICKET_PL),
    ("formal", "Szanowny Panie, uprzejmie proszę o przesłanie faktury do końca tygodnia."),
    ("long", "Chociaż pociąg z Krakowa spóźnił się o ponad godzinę, a na dworcu panował straszny tłok, "
             "udało nam się zdążyć na spotkanie, które zaczynało się o dziesiątej."),
    ("numbers", "Umowa została podpisana 3 marca 2024 r., a jej wartość wynosi 12 500 zł netto."),
    ("names", "Maria Skłodowska-Curie urodziła się w Warszawie i pracowała na Sorbonie w Paryżu."),
    ("idiom", "Nie mój cyrk, nie moje małpy."),
]

# Decoding settings shared by both runtimes. Spark NLP's M2M100/NLLB defaults (minOutputLength=10,
# noRepeatNgramSize=3) and each HF model's generation_config (beam search, etc.) would otherwise differ.
HF_OVERRIDES = {"num_beams": 1, "do_sample": False, "min_length": 0, "no_repeat_ngram_size": 0,
                "repetition_penalty": 1.0, "max_new_tokens": MAX_OUT}


# ---- run config -------------------------------------------------------------------------------------------

def set_smoke(flag, n_full=N_FULL):
    """SMOKE runs N_SMOKE sentences per model, the full pass n_full; saved to a file so it survives restarts."""
    os.makedirs(ROOT, exist_ok=True)
    pathlib.Path(CONFIG_JSON).write_text(json.dumps({"smoke": bool(flag), "n_full": n_full}))


def n_sentences():
    if not os.path.exists(CONFIG_JSON):
        return N_FULL
    config = json.loads(pathlib.Path(CONFIG_JSON).read_text())
    return N_SMOKE if config.get("smoke") else config.get("n_full", N_FULL)


# ---- data -------------------------------------------------------------------------------------------------

def read_flores_pair(pl_path, de_path, n):
    pl = pathlib.Path(pl_path).read_text(encoding="utf-8").splitlines()
    de = pathlib.Path(de_path).read_text(encoding="utf-8").splitlines()
    if len(pl) != len(de):
        raise ValueError(f"{pl_path} has {len(pl)} lines but {de_path} has {len(de)}")
    return [(i, pl[i], de[i]) for i in range(min(n, len(pl)))]


def _flores_from_tarball(n):
    import tarfile

    folder = os.path.join(ROOT, "flores")
    os.makedirs(folder, exist_ok=True)
    archive = os.path.join(folder, "flores200_dataset.tar.gz")
    if not os.path.exists(archive):
        urllib.request.urlretrieve(FLORES_URL, archive)
    paths = []
    with tarfile.open(archive) as tar:
        names = tar.getnames()  # the published archive stores them as ./flores200_dataset/...
        for lang in ("pol_Latn", "deu_Latn"):
            member = next(n for n in names if n.endswith(f"flores200_dataset/devtest/{lang}.devtest"))
            out = os.path.join(folder, f"{lang}.devtest")
            pathlib.Path(out).write_bytes(tar.extractfile(member).read())
            paths.append(out)
    return read_flores_pair(paths[0], paths[1], n)


def _flores_from_hub(n):
    from datasets import load_dataset

    folder = os.path.join(ROOT, "flores")
    os.makedirs(folder, exist_ok=True)
    paths = []
    for lang in ("pol_Latn", "deu_Latn"):
        rows = load_dataset("openlanguagedata/flores_plus", lang, split="devtest", token=os.environ.get("HF_TOKEN"))
        out = os.path.join(folder, f"{lang}.devtest")
        pathlib.Path(out).write_text("\n".join(rows["text"]) + "\n", encoding="utf-8")
        paths.append(out)
    return read_flores_pair(paths[0], paths[1], n)


def prepare_flores(n=N_FULL):
    """Writes the first n FLORES-200 devtest pairs (Polish source, German reference) to DATA_TSV."""
    os.makedirs(ROOT, exist_ok=True)
    try:
        pairs, source = _flores_from_tarball(n), FLORES_URL
    except Exception as e:  # the tarball has moved before; flores_plus is gated and needs HF_TOKEN
        print(f"FLORES tarball unavailable ({e}); falling back to openlanguagedata/flores_plus")
        pairs, source = _flores_from_hub(n), "openlanguagedata/flores_plus"
    with open(DATA_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "pl", "de_ref"])
        w.writerows(pairs)
    print(f"{len(pairs)} sentence pairs from {source} -> {DATA_TSV}")
    return DATA_TSV


def load_sentences(n=None):
    """The first n sentence pairs (default: this pass's count, set by SMOKE)."""
    with open(DATA_TSV, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    return [(int(i), pl, de) for i, pl, de in rows[:n or n_sentences()]]


# ---- fair settings ----------------------------------------------------------------------------------------

def spark_settings(key, pad_token_id=None):
    """Annotator setter name (without "set") -> value. Every value mirrors HF_OVERRIDES."""
    s = {"batchSize": MODELS[key]["batch"], "doSample": HF_OVERRIDES["do_sample"],
         "maxOutputLength": HF_OVERRIDES["max_new_tokens"],
         "noRepeatNgramSize": HF_OVERRIDES["no_repeat_ngram_size"],
         "repetitionPenalty": HF_OVERRIDES["repetition_penalty"]}
    if key == "marian":  # greedy only, no minimum-length setting; input cap defaults to 40 tokens
        s["maxInputLength"] = MODELS[key]["max_input"]
        if pad_token_id is not None:  # HF Marian bans the pad token through bad_words_ids
            s["ignoreTokenIds"] = [pad_token_id]
    else:  # M2M100/NLLB input cap is fixed at 1024 with no Python setter
        s.update(beamSize=HF_OVERRIDES["num_beams"], minOutputLength=HF_OVERRIDES["min_length"],
                 srcLang=MODELS[key]["src"], tgtLang=MODELS[key]["tgt"])
    return s


def apply_spark_settings(annotator, settings):
    for name, value in settings.items():
        annotator = getattr(annotator, "set" + name[0].upper() + name[1:])(value)
    return annotator


def hf_generation_config(model, tokenizer, key):
    cfg = copy.deepcopy(model.generation_config)
    cfg.update(**HF_OVERRIDES)
    if key == "m2m100":
        cfg.forced_bos_token_id = tokenizer.get_lang_id("de")
    elif key == "nllb":
        cfg.forced_bos_token_id = tokenizer.convert_tokens_to_ids(MODELS[key]["tgt"])
    return cfg  # Marian keeps its bad_words_ids=[[pad]]


def hf_pad_token_id(key):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(MODELS[key]["hf_id"]).pad_token_id


def hf_precision(model, key):
    """Brings the Hugging Face model to the Spark NLP side's precision. int8: PyTorch's dynamic quantization of the
    Linear layers (int8 weights, activations quantized on the fly), in place so the fp32 weights aren't kept twice."""
    if MODELS[key]["precision"] != "int8":
        return model
    import torch

    engines = torch.backends.quantized.supported_engines
    if "fbgemm" not in engines and "qnnpack" in engines:  # ARM builds (local rehearsal); Colab's x86 has fbgemm
        torch.backends.quantized.engine = "qnnpack"
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


def load_hf(key):
    """The Hugging Face model and tokenizer at MODELS[key]["precision"]; quantizing is part of the load."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    hf_id = MODELS[key]["hf_id"]
    return hf_precision(AutoModelForSeq2SeqLM.from_pretrained(hf_id).eval(), key), AutoTokenizer.from_pretrained(hf_id)


# ---- session ----------------------------------------------------------------------------------------------

def stop_spark():
    """Stop Spark and end its Java process, so the next start_spark() gets a new one.

    A new JVM frees the old model's memory, reads spark.jsl.settings again, and copies model files into a new folder
    (every export ships an encoder_model.onnx, and one JVM refuses a second, different file with that name).
    """
    from pyspark import SparkContext
    from pyspark.sql import SparkSession

    if SparkSession._instantiatedSession is not None:
        SparkSession._instantiatedSession.stop()
    elif SparkContext._active_spark_context is not None:
        SparkContext._active_spark_context.stop()
    gateway = SparkContext._gateway
    if gateway is None:
        return
    gateway.shutdown()
    proc = getattr(gateway, "proc", None)
    if proc is not None:
        proc.stdin.close()  # pyspark's JVM exits when its stdin closes
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
            proc.wait()
    SparkContext._gateway = SparkContext._jvm = None  # the next SparkContext launches a new JVM


def start_spark():
    """Spark NLP with 1 ONNX thread per task; local[*] gives 2 tasks on Colab's 2 vCPUs, matching torch's 2.

    Any Spark already running in this kernel is stopped first, with its Java process, so a cell run twice starts clean.
    """
    import sparknlp

    stop_spark()
    params = {"spark.jsl.settings.onnx.intraOpNumThreads": "1"}
    jar = os.environ.get("SN1455_JAR")  # local rehearsal only
    if jar:
        params["spark.jars"] = jar
        return sparknlp.start(memory=SPARK_MEMORY, params=params, skip_sparknlp_maven=True)
    return sparknlp.start(memory=SPARK_MEMORY, params=params)


def config_summary(spark):
    import torch

    sc = spark.sparkContext
    return {"master": sc.master, "default_parallelism": sc.defaultParallelism,
            "onnx_intra_op_threads": spark.conf.get("spark.jsl.settings.onnx.intraOpNumThreads", None),
            "driver_memory": spark.conf.get("spark.driver.memory", None),
            "torch_threads": torch.get_num_threads(), "n_sentences": n_sentences()}


# ---- Spark NLP models and pipelines -----------------------------------------------------------------------

def aligned_batches(partition_ids, size):
    """The batches Spark NLP forms: consecutive groups of `size` inside each partition, never across two."""
    return [ids[i:i + size] for ids in partition_ids for i in range(0, len(ids), size)]


def spark_partition_ids(df):
    """The ids in each non-empty partition, in order. Uses spark_partition_id(), which runs in the JVM: Colab runs
    Python 3.13 and pyspark 3.5 supports up to 3.11, so nothing here may start a Python worker (no df.rdd, no UDFs)."""
    from pyspark.sql.functions import spark_partition_id

    parts = {}
    for r in df.select("id", spark_partition_id().alias("partition")).collect():
        parts.setdefault(r["partition"], []).append(r["id"])
    return [parts[p] for p in sorted(parts)]


def join_annotations(results):
    """One input can come back as several annotations (e.g. one per sentence); keep them all."""
    return " ".join(r.strip() for r in results if r and r.strip())


def saved_model_path(key):
    return os.path.join(MODELS_DIR, f"{key}_spark_nlp")


def _annotator_class(key):
    import sparknlp.annotator

    return getattr(sparknlp.annotator, MODELS[key]["annotator"])


def load_exported(key, export_dir, spark):
    return _annotator_class(key).loadSavedModel(export_dir, spark)


def load_saved(key):
    return _annotator_class(key).load(saved_model_path(key))


def save_spark_model(annotator, key):
    path = saved_model_path(key)
    annotator.write().overwrite().save(path)
    return path


def build_pipeline(key, annotator, settings, spark):
    from pyspark.ml import Pipeline
    from sparknlp.base import DocumentAssembler

    document = DocumentAssembler().setInputCol("text").setOutputCol("document")
    translator = apply_spark_settings(annotator.setInputCols(["document"]).setOutputCol("translation"), settings)
    return Pipeline(stages=[document, translator]).fit(spark.createDataFrame([], "id INT, text STRING"))


def check_translation(pipeline_model, spark, text):
    outputs, _, _ = run_spark_batch(pipeline_model, spark, [(0, text)])
    return outputs[0]


def _require(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found in {folder}")
    return path


def fetch_int8_onnx(key, export_dir):
    """Lays out export_dir as an export would: the int8 encoder/decoder from MODELS[key]["onnx"] under the names
    loadSavedModel reads, plus the original checkpoint's config and tokenizer files. Also caches the checkpoint's
    weights, which the Hugging Face side loads."""
    import shutil

    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(export_dir, exist_ok=True)
    repo, revision = MODELS[key]["onnx"]
    for name, remote in INT8_FILES.items():
        shutil.copy(hf_hub_download(repo, remote, revision=revision), os.path.join(export_dir, name))
    checkpoint = snapshot_download(MODELS[key]["hf_id"], allow_patterns=["*.json", "*.model", "pytorch_model.bin"])
    for name in ("config.json", "sentencepiece.bpe.model", "vocab.json"):  # NLLB has no vocab.json
        if os.path.exists(os.path.join(checkpoint, name)):
            shutil.copy(os.path.join(checkpoint, name), os.path.join(export_dir, name))
    return sorted(os.listdir(export_dir))


def prepare_assets(key, export_dir, vocab=None):
    """Lays out export_dir/assets the way <Annotator>.loadSavedModel reads it, as in the repo's export notebooks."""
    import shutil

    assets = os.path.join(export_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    if key == "marian":
        for name in ("source.spm", "target.spm"):
            shutil.move(_require(export_dir, name), os.path.join(assets, name))
        tokens = json.loads(pathlib.Path(_require(export_dir, "vocab.json")).read_text(encoding="utf-8"))
        lines = sorted(tokens, key=tokens.get)  # the loader numbers tokens by line
        pathlib.Path(assets, "vocab.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        config = json.loads(pathlib.Path(_require(export_dir, "config.json")).read_text(encoding="utf-8"))
        for name in ("begin_suppress_tokens", "suppress_tokens"):  # loadSavedModel reads both as int arrays
            config[name] = config.get(name) or []
        spm = _require(export_dir, "sentencepiece.bpe.model")
        if key == "nllb":  # the checkpoint has no vocab.json; build it from the tokenizer
            if vocab is None:
                from transformers import AutoTokenizer

                vocab = AutoTokenizer.from_pretrained(MODELS[key]["hf_id"]).get_vocab()
        else:
            vocab = json.loads(pathlib.Path(_require(export_dir, "vocab.json")).read_text(encoding="utf-8"))
        pathlib.Path(assets, "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
        shutil.copy(spm, os.path.join(assets, "sentencepiece.bpe.model"))
        pathlib.Path(assets, "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")  # escaped: JVM charset-safe
    return sorted(os.listdir(assets))


# ---- runners: text in -> text out, timed the same way on both sides ---------------------------------------

def error_summary(e):
    """One line for a failure: the root "Caused by" of a Java error raised through py4j, else the first line."""
    causes = [line.split("Caused by:", 1)[1].strip() for line in str(e).splitlines() if line.startswith("Caused by:")]
    return causes[-1] if causes else f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _spark_outputs(collected):
    return {r["id"]: join_annotations(r["result"] or []) for r in collected}


def run_spark_batch(pipeline_model, spark, rows):
    """The default Pipeline over one DataFrame, timed from text in to text out (createDataFrame included, as in
    run_spark_single). Also returns the partition layout it used."""
    def run():
        df = spark.createDataFrame(list(rows), ["id", "text"])
        return df, pipeline_model.transform(df).select("id", "translation.result").collect()

    (df, collected), secs = timed(run)
    return _spark_outputs(collected), secs, spark_partition_ids(df)


def run_spark_single(pipeline_model, spark, rows):
    """One Spark job per sentence, so Spark's per-job overhead is part of the latency."""
    outputs, latencies = {}, {}
    for i, text in rows:
        collected, latencies[i] = timed(lambda: pipeline_model.transform(
            spark.createDataFrame([(i, text)], ["id", "text"])).select("id", "translation.result").collect())
        outputs.update(_spark_outputs(collected))
    return outputs, latencies


def _hf_translate(model, tokenizer, gen_config, key, texts):
    import torch

    if MODELS[key]["src"]:
        tokenizer.src_lang = MODELS[key]["src"]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MODELS[key]["max_input"])
    with torch.no_grad():
        generated = model.generate(**enc, generation_config=gen_config)
    return [t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def run_hf_batch(model, tokenizer, gen_config, key, rows, groups):
    text = dict(rows)
    outputs = {}

    def run():
        for g in groups:
            outputs.update(zip(g, _hf_translate(model, tokenizer, gen_config, key, [text[i] for i in g])))

    _, secs = timed(run)
    return outputs, secs


def run_hf_single(model, tokenizer, gen_config, key, rows):
    outputs, latencies = {}, {}
    for i, text in rows:
        (outputs[i],), latencies[i] = timed(lambda: _hf_translate(model, tokenizer, gen_config, key, [text]))
    return outputs, latencies


# ---- CPU and memory sampling ------------------------------------------------------------------------------

def find_java_pid():
    import psutil

    for p in psutil.Process().children(recursive=True):
        try:
            if "java" in p.name().lower():
                return p.pid
        except psutil.Error:
            pass
    return None


class CpuSampler:
    """Every `interval` s: per-core CPU %, java and python RSS, java thread count, tagged with the current phase."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.rows = []
        self._phase = "idle"
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        import psutil

        psutil.cpu_percent(percpu=True)  # the first call only primes the counters
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    @contextlib.contextmanager
    def phase(self, label):
        previous, self._phase = self._phase, label
        try:
            yield
        finally:
            self._phase = previous

    def _run(self):
        import psutil

        me, java = psutil.Process(), None
        while not self._stop.wait(self.interval):
            if java is None:
                pid = find_java_pid()
                java = psutil.Process(pid) if pid else None
            row = {"t": round(time.perf_counter() - self._t0, 3), "phase": self._phase,
                   "py_rss_mb": me.memory_info().rss / 2**20}
            row.update({f"cpu_core{c}": pct for c, pct in enumerate(psutil.cpu_percent(percpu=True))})
            try:
                row["java_rss_mb"] = java.memory_info().rss / 2**20 if java else float("nan")
                row["java_threads"] = java.num_threads() if java else float("nan")
            except psutil.Error:
                java, row["java_rss_mb"], row["java_threads"] = None, float("nan"), float("nan")
            self.rows.append(row)

    def frame(self):
        import pandas as pd
        import psutil

        cores = [f"cpu_core{c}" for c in range(psutil.cpu_count())]
        return pd.DataFrame(self.rows, columns=["t", "phase", *cores, "java_rss_mb", "py_rss_mb", "java_threads"])


def peak(frame, phase_prefix, column):
    return round(float(frame[frame.phase.str.startswith(phase_prefix)][column].max()), 1)


# ---- memory per sentence (the M2M100 leak notebook) -------------------------------------------------------

def available_gb():
    import psutil

    return psutil.virtual_memory().available / 2**30


def java_memory_mb(spark):
    """The Java process's resident memory and its heap. What the heap doesn't hold is native memory, where ONNX
    Runtime keeps its tensors."""
    import psutil

    runtime = spark._jvm.java.lang.Runtime.getRuntime()
    committed, free = runtime.totalMemory(), runtime.freeMemory()
    return {"java_rss_mb": round(psutil.Process(find_java_pid()).memory_info().rss / 2**20),
            "heap_committed_mb": round(committed / 2**20), "heap_used_mb": round((committed - free) / 2**20)}


def predicted_m2m100_leak_mb(out_tokens, vocab_size=128112):
    """What one translation leaves behind if M2M100.getDecoderOutputs never closes its per-step outputs: the decoder
    starts from 2 tokens (end-of-sentence, target language) and runs once per new token until end-of-sentence, so n
    output tokens take n+1 runs over 2..n+2 positions, each returning vocab_size fp32 scores per position."""
    positions = sum(range(2, out_tokens + 3))
    return round(positions * vocab_size * 4 / 2**20, 1)


def spark_memory_per_sentence(pipeline_model, spark, rows, tokenizer, key="m2m100", safety_gb=1.5):
    """Translates rows one Spark job at a time and reads the Java process's memory after each. Stops early when the
    machine has under safety_gb left: past that, Colab's out-of-memory killer ends Java and the readings are lost."""
    import pandas as pd

    records, before = [], java_memory_mb(spark)["java_rss_mb"]
    for n, (i, text) in enumerate(rows, 1):
        outputs, latencies = run_spark_single(pipeline_model, spark, [(i, text)])
        memory, tokens = java_memory_mb(spark), hf_token_count(tokenizer, outputs.get(i, ""), key)
        records.append({"n": n, "id": i, "seconds": round(latencies[i], 1), "out_tokens": tokens, **memory,
                        "growth_mb": memory["java_rss_mb"] - before,
                        "predicted_leak_mb": predicted_m2m100_leak_mb(tokens), "output": outputs.get(i, "")})
        before = memory["java_rss_mb"]
        r = records[-1]
        print(f"{n:2d}: {tokens:3d} tokens in {r['seconds']:5.1f} s | Java {r['java_rss_mb']} MB (+{r['growth_mb']}, "
              f"predicted +{r['predicted_leak_mb']:.0f}) | heap used {r['heap_used_mb']} MB", flush=True)
        if available_gb() < safety_gb:
            print(f"stopped after {n} of {len(rows)} sentences: under {safety_gb} GB of memory left on this machine")
            break
    return pd.DataFrame(records)


def hf_memory_per_sentence(model, tokenizer, gen_config, key, rows):
    """Translates rows one at a time with Hugging Face and reads this Python process's memory after each."""
    import pandas as pd
    import psutil

    records, me = [], psutil.Process()
    for n, (i, text) in enumerate(rows, 1):
        outputs, latencies = run_hf_single(model, tokenizer, gen_config, key, [(i, text)])
        records.append({"n": n, "id": i, "seconds": round(latencies[i], 1),
                        "out_tokens": hf_token_count(tokenizer, outputs[i], key),
                        "py_rss_mb": round(me.memory_info().rss / 2**20), "output": outputs[i]})
    return pd.DataFrame(records)


# ---- metrics ----------------------------------------------------------------------------------------------

def score(hyps, refs):
    import sacrebleu

    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}


def latency_stats(latencies):
    import numpy as np

    return {"latency_median_s": round(float(np.percentile(latencies, 50)), 4),
            "latency_p95_s": round(float(np.percentile(latencies, 95)), 4)}


def exact_match_pct(a, b):
    return round(100 * sum(x.strip() == y.strip() for x, y in zip(a, b)) / len(a), 2)


def hf_token_count(tokenizer, text, key):
    """Output length in the model's own target-side tokens (Marian has separate source and target vocabularies).
    M2M100/NLLB tokenizers ship tgt_lang=None and fail in target mode until it is set."""
    if MODELS[key]["tgt"]:
        tokenizer.tgt_lang = MODELS[key]["tgt"]
    return len(tokenizer(text_target=text, add_special_tokens=False).input_ids)


def count_at_cap(token_counts, cap=MAX_OUT):
    return sum(c >= cap for c in token_counts)


def dir_size_mb(path):
    return round(sum(f.stat().st_size for f in pathlib.Path(path).rglob("*") if f.is_file()) / 2**20, 1)


def hf_weights_mb(hf_id):
    from huggingface_hub import try_to_load_from_cache

    for name in ("model.safetensors", "pytorch_model.bin"):  # from_pretrained prefers safetensors
        path = try_to_load_from_cache(hf_id, name)
        if isinstance(path, str):
            return round(os.path.getsize(path) / 2**20, 1)
    return float("nan")


def runtime_metrics(outputs, refs, load_s, disk_mb, single_lat, batch_s, peak_rss_mb):
    ids = sorted(outputs)
    m = score([outputs[i] for i in ids], [refs[i] for i in ids])
    m.update(load_s=round(load_s, 2), disk_mb=disk_mb, **latency_stats(list(single_lat.values())),
             throughput_sps=round(len(ids) / batch_s, 3), peak_rss_mb=peak_rss_mb)
    return m


# ---- results ----------------------------------------------------------------------------------------------

def _result_path(key, suffix):
    return os.path.join(RESULTS_DIR, f"{key}_{suffix}")


def save_results(key, outputs, cpu, metrics, qualitative, single):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for frame, suffix in ((outputs, "outputs"), (cpu, "cpu"), (qualitative, "qualitative"), (single, "single")):
        frame.to_csv(_result_path(key, f"{suffix}.csv"), index=False)
    pathlib.Path(_result_path(key, "metrics.json")).write_text(json.dumps(metrics, indent=1, ensure_ascii=False))


def save_export_info(key, info):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pathlib.Path(_result_path(key, "export.json")).write_text(json.dumps(info, indent=1, ensure_ascii=False))


def load_all_results():
    """Every model whose benchmark finished in this pass, in ORDER; a missing model is skipped, not an error."""
    import pandas as pd

    results = {}
    for key in ORDER:
        if not os.path.exists(_result_path(key, "metrics.json")):
            continue
        metrics = json.loads(pathlib.Path(_result_path(key, "metrics.json")).read_text())
        n = metrics.get("config", {}).get("n_sentences")
        if n is not None and n != n_sentences():  # left over from the other pass (SMOKE vs full)
            print(f"skipping {key}: its saved results are from a {n}-sentence run, this pass uses {n_sentences()}")
            continue
        export = _result_path(key, "export.json")
        results[key] = {
            "metrics": metrics,
            **{s: pd.read_csv(_result_path(key, f"{s}.csv")) for s in ("cpu", "outputs", "qualitative", "single")},
            "export": json.loads(pathlib.Path(export).read_text()) if os.path.exists(export) else None,
        }
    return results
