from sentence_transformers import SentenceTransformer, util
from evaluate import load as eval_load
from torcheval.metrics import WordErrorRate


# Detects hallucinations based on https://arxiv.org/html/2401.01572v1
def detect_hallucinations_article(y_true, y_pred, thr_wer, thr_cs, thr_per, verbose=False):
    assert(len(y_true) == len(y_pred))

    # Prepare models for metrics calculations
    model_cos_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    perplex_metric = eval_load("perplexity", module_type="metric")

    # Count potential hallucinations
    hallucinations = 0
    for i in range(len(y_true)):
        wer_metric = WordErrorRate()
        wer_metric.update(y_true[i], y_pred[i])
        wer = wer_metric.compute()

        if wer > thr_wer:
            embedded_true = model_cos_sim.encode(y_true[i], convert_to_tensor=True)
            embedded_pred = model_cos_sim.encode(y_pred[i], convert_to_tensor=True)
            cos_sim = util.pytorch_cos_sim(embedded_true, embedded_pred)

            _perplexity = perplex_metric.compute(predictions=[y_pred[i]], model_id="gpt2")
            perplexity = _perplexity["perplexities"][0]

            if cos_sim < thr_cs and perplexity < thr_per:
                hallucinations += 1
                if verbose:
                    print("True:", y_true[i])
                    print("Pred:", y_pred[i])

    if verbose:
        print("Total sencences:", len(y_true), "\nPotentially hallucinatory sentences:", hallucinations)

    return {"total_sentences": len(y_true), "potentially_hallucinatory_sentences": hallucinations}


# Our simple hallucination detection based on our empirical observations
def detect_hallucinations_simple(y_true, y_pred, verbose=False):
    assert(len(y_true) == len(y_pred))

    # Prepare model for cosine similarity calculation
    model_cos_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Count potential hallucinations
    hallucinations = 0
    for i in range(len(y_true)):
        wer_metric = WordErrorRate()
        wer_metric.update(y_true[i], y_pred[i])
        wer = wer_metric.compute()

        embedded_true = model_cos_sim.encode(y_true[i], convert_to_tensor=True)
        embedded_pred = model_cos_sim.encode(y_pred[i], convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embedded_true, embedded_pred)

        if len(y_pred[i]) > len(y_true[i]) and cos_sim > (1.0 - wer):
            hallucinations += 1
            if verbose:
                print("True:", y_true[i])
                print("Pred:", y_pred[i])
                print("WER:", wer, "Cosine similarity:", cos_sim.item())
                print("\n")

    if verbose:
        print("Total sencences:", len(y_true), "\nPotentially hallucinatory sentences:", hallucinations)

    return {"total_sentences": len(y_true), "potentially_hallucinatory_sentences": hallucinations}