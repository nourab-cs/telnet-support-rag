#!/usr/bin/env python3
"""
evaluate_rag.py
Évalue automatiquement le RAG TELNET SmartConnect.

Ce script :
- charge le jeu de tests JSON ;
- instancie le même RAG que main.py (configuration actuelle) ;
- exécute les questions dans l'ordre ;
- conserve une même session pour les tests marqués avec le même "session" ;
- calcule Hit@K, Recall@K, Precision@K, MRR au niveau source ;
- mesure les latences retrieval / génération / totale ;
- vérifie le déclenchement attendu du retrieval ;
- calcule un indicateur lexical simple sur answer_keywords ;
- sauvegarde les résultats détaillés en JSON et les métriques agrégées en CSV.

IMPORTANT :
Les métriques lexicales ne remplacent pas Faithfulness / Answer Relevancy.
Pour une première évaluation reproductible, ce script privilégie des métriques
déterministes et directement observables dans la trace du pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


JUDGE_SYSTEM_PROMPT = """Tu es un évaluateur expert de systèmes RAG. Évalue la réponse à partir du contexte fourni. Retourne UNIQUEMENT un JSON valide avec les champs faithfulness, answer_relevancy, context_relevancy, correctness (0-1), overall_score (0-5) et reason (court, français).

Faithfulness: la réponse est-elle supportée par le contexte ?
Answer Relevancy: répond-elle directement et précisément à la question ?
Context Relevancy: les documents récupérés sont-ils utiles pour répondre ?
Correctness: est-elle correcte selon le contexte, expected_behavior et answer_keywords ?
Overall_score: note globale 0-5 (5 excellent, 0 inutilisable).
"""


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    # Keep accents but normalize whitespace/punctuation enough for keyword tests.
    text = re.sub(r"\s+", " ", text)
    return text


def basename(value: Any) -> str:
    if not value:
        return ""
    return os.path.basename(str(value).replace("\\", "/"))


def source_matches(actual_source: str, expected_source: str) -> bool:
    """
    Matching souple :
    - exact basename ;
    - expected_source peut être un nom partiel comme '03_documentation_api'.
    """
    a = normalize_text(basename(actual_source))
    e = normalize_text(expected_source)
    return a == e or e in a or a in e


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (rank - low)


def extract_sources(result: Dict[str, Any]) -> List[str]:
    docs = result.get("documents") or result.get("source_documents") or []
    sources = []
    for doc in docs:
        metadata = getattr(doc, "metadata", None) or {}
        source = metadata.get("source") or metadata.get("relative_path") \
            or metadata.get("file_path") or metadata.get("filename")
        if source:
            sources.append(str(source))
    return sources


def extract_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    trace = result.get("trace")
    return trace if isinstance(trace, dict) else {}


def trace_retrieval_executed(trace: Dict[str, Any], result: Dict[str, Any]) -> bool:
    retrieval = trace.get("retrieval")
    if isinstance(retrieval, dict) and "executed" in retrieval:
        return bool(retrieval["executed"])
    return bool(result.get("retrieved_count", 0))


def trace_latencies(result: Dict[str, Any]) -> Tuple[float, float, float]:
    trace = extract_trace(result)
    latency = trace.get("latency") or {}
    total = float(latency.get("total") or 0.0)
    retrieval = float(latency.get("retrieval") or 0.0)
    generation = float(latency.get("generation") or 0.0)
    return retrieval, generation, total


def keyword_score(answer: str, keywords: Sequence[str]) -> Optional[float]:
    if not keywords:
        return None
    text = normalize_text(answer)
    hits = sum(1 for kw in keywords if normalize_text(kw) in text)
    return hits / len(keywords)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
    return str(content or "")


def _parse_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise ValueError(f"Réponse du juge non JSON: {text[:300]}")
        obj = json.loads(text[a:b+1])
    if not isinstance(obj, dict):
        raise ValueError("Le juge n'a pas retourné un objet JSON")
    return obj


def _score01(x: Any) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return None


def _score05(x: Any) -> Optional[float]:
    try:
        return max(0.0, min(5.0, float(x)))
    except (TypeError, ValueError):
        return None


def build_judge_context(documents: Sequence[Any], max_chars: int = 18000) -> str:
    blocks, total = [], 0
    for i, doc in enumerate(documents, 1):
        meta = getattr(doc, "metadata", None) or {}
        source = meta.get("source") or meta.get("relative_path") or meta.get("file_path") or meta.get("filename") or "source_inconnue"
        text = str(getattr(doc, "page_content", "") or "")
        block = f"[DOCUMENT {i} — {basename(source)}]\n{text}"
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining] + "\n[... contexte tronqué ...]"
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def evaluate_with_llm_judge(judge_llm: Any, question: str, answer: str, context: str, expected_behavior: str, keywords: Sequence[str]) -> Dict[str, Any]:
    prompt = f"""QUESTION:\n{question}\n\nRÉPONSE DU RAG:\n{answer}\n\nCONTEXTE RÉCUPÉRÉ:\n{context or '[Aucun contexte]'}\n\nRÉFÉRENCE DATASET:\nexpected_behavior = {expected_behavior or '[non renseigné]'}\nanswer_keywords = {', '.join(map(str, keywords)) or '[aucun]'}\n\nRetourne le JSON demandé."""
    try:
        response = judge_llm.invoke([("system", JUDGE_SYSTEM_PROMPT), ("human", prompt)])
    except Exception:
        # Fallback utile avec certaines versions de langchain-ollama/Ollama.
        response = judge_llm.invoke(JUDGE_SYSTEM_PROMPT + "\n\n" + prompt)
    raw = _response_text(response)
    obj = _parse_json(raw)
    return {
        "faithfulness": _score01(obj.get("faithfulness")),
        "answer_relevancy": _score01(obj.get("answer_relevancy")),
        "context_relevancy": _score01(obj.get("context_relevancy")),
        "correctness": _score01(obj.get("correctness")),
        "overall_score": _score05(obj.get("overall_score")),
        "reason": str(obj.get("reason") or "").strip(),
        "raw_response": raw,
    }


def create_judge(model_name: str):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model_name,
        temperature=0.0,
        num_predict=512,
        format="json",
    )


def retrieval_metrics(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k: int,
) -> Dict[str, Optional[float]]:
    """
    Métriques au niveau source.
    Une source attendue est considérée retrouvée si au moins un chunk de cette
    source apparaît dans le top-k.
    """
    expected = list(dict.fromkeys(expected_sources))
    if not expected:
        return {
            "hit_at_k": None,
            "recall_at_k": None,
            "precision_at_k": None,
            "mrr": None,
        }

    topk = list(retrieved_sources[:k])

    matched_expected = {
        exp for exp in expected
        if any(source_matches(src, exp) for src in topk)
    }

    hit = 1.0 if matched_expected else 0.0
    recall = len(matched_expected) / len(expected)

    # Precision source-level : nombre de sources distinctes retrouvées
    # qui appartiennent à l'ensemble attendu / nombre de sources distinctes top-k.
    distinct_topk = []
    for src in topk:
        if not any(source_matches(src, x) for x in distinct_topk):
            distinct_topk.append(src)

    relevant_distinct = sum(
        1 for src in distinct_topk
        if any(source_matches(src, exp) for exp in expected)
    )
    precision = relevant_distinct / len(distinct_topk) if distinct_topk else 0.0

    rr = 0.0
    for rank, src in enumerate(topk, start=1):
        if any(source_matches(src, exp) for exp in expected):
            rr = 1.0 / rank
            break

    return {
        "hit_at_k": hit,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "mrr": rr,
    }


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def load_pipeline_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_pipeline(pipeline_class):
    """
    Même configuration que celle utilisée dans main.py.
    Ne modifie pas les paramètres du RAG.
    """
    return pipeline_class(
        data_dir=str(PROJECT_ROOT / "data"),
        db_dir=str(PROJECT_ROOT / "chroma_db"),
        collection_name="telnet_support",

        embedding_model="BAAI/bge-m3",
        llm_model="mistral",

        retrieval_type="hybrid",
        retrieval_k=8,
        retrieval_fetch_k=20,
        retrieval_lambda=0.6,
        hybrid_k=10,

        relevance_threshold=None,
        similarity_score_threshold=0.40,

        max_context_documents=6,

        use_history=True,
        max_history=10,
        max_history_chars=5000,

        enable_query_rewriting=True,
    )


def load_or_build_index(pipeline) -> str:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.load_existing()
        return "existing"
    except (FileNotFoundError, ValueError):
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.build_index()
        return "built"


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def run_evaluation(
    pipeline,
    tests: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    verbose: bool = False,
    judge_llm: Any = None,
    judge_context_chars: int = 18000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []
    previous_session = None

    for position, test in enumerate(tests, start=1):
        test_id = test.get("id", position)
        question = str(test.get("question", ""))
        category = str(test.get("category", ""))
        expected_behavior = str(test.get("expected_behavior", ""))
        evaluation = test.get("eval") or {}

        session = evaluation.get("session") or f"test_{test_id}"

        # Même session pour les tests historiques partageant le même identifiant.
        # Sinon, on nettoie l'historique pour éviter la contamination entre tests.
        if previous_session != session:
            if hasattr(pipeline, "clear_history"):
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.clear_history()
        previous_session = session

        if verbose:
            print(f"\n[{position}/{len(tests)}] Test {test_id} — {question}")

        start = time.perf_counter()

        # Le pipeline de traçabilité affiche volontairement beaucoup d'informations.
        # On les capture par défaut pour garder le benchmark lisible.
        trace_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(trace_buffer):
                result = pipeline.ask(question)
            run_error = None
        except Exception as exc:
            result = {}
            run_error = f"{type(exc).__name__}: {exc}"

        wall_latency = time.perf_counter() - start

        if not isinstance(result, dict):
            result = {"answer": str(result)}

        answer = str(result.get("answer") or "")
        trace = extract_trace(result)
        retrieved_sources = extract_sources(result)

        retrieval_executed = trace_retrieval_executed(trace, result)
        retrieval_latency, generation_latency, total_latency = trace_latencies(result)

        # Si la trace n'est pas disponible, le temps mur est un fallback.
        if total_latency <= 0:
            total_latency = wall_latency

        expected_retrieval = evaluation.get("expected_retrieval")
        retrieval_behavior_ok = None
        if expected_retrieval is not None:
            retrieval_behavior_ok = bool(expected_retrieval) == retrieval_executed

        expected_sources = evaluation.get("expected_sources") or []
        answer_keywords = evaluation.get("answer_keywords") or []

        metrics_by_k = {}
        for k in k_values:
            metrics_by_k[str(k)] = retrieval_metrics(
                retrieved_sources,
                expected_sources,
                k,
            )

        primary_k = max(k_values) if k_values else 5
        primary = metrics_by_k.get(str(primary_k), {})

        kw_score = keyword_score(answer, answer_keywords)

        llm_judge = {
            "enabled": judge_llm is not None,
            "faithfulness": None,
            "answer_relevancy": None,
            "context_relevancy": None,
            "correctness": None,
            "overall_score": None,
            "reason": None,
            "latency_s": None,
            "error": None,
        }
        if judge_llm is not None and not run_error:
            t_judge = time.perf_counter()
            try:
                judged = evaluate_with_llm_judge(
                    judge_llm, question, answer,
                    build_judge_context(result.get("documents") or result.get("source_documents") or [], judge_context_chars),
                    expected_behavior, answer_keywords,
                )
                llm_judge.update(judged)
            except Exception as exc:
                llm_judge["error"] = f"{type(exc).__name__}: {exc}"
                # Conserve un aperçu de l'erreur dans les résultats détaillés.
                llm_judge["reason"] = "Échec de l'évaluation LLM : " + str(exc)[:500]
            llm_judge["latency_s"] = time.perf_counter() - t_judge

        row = {
            "id": test_id,
            "position": position,
            "category": category,
            "session": session,
            "question": question,
            "expected_behavior": expected_behavior,
            "expected_retrieval": expected_retrieval,
            "retrieval_executed": retrieval_executed,
            "retrieval_behavior_ok": retrieval_behavior_ok,
            "expected_sources": expected_sources,
            "retrieved_sources": [basename(x) for x in retrieved_sources],
            "answer": answer,
            "answer_keywords": answer_keywords,
            "answer_keyword_score": kw_score,
            "hit_at_k": primary.get("hit_at_k"),
            "recall_at_k": primary.get("recall_at_k"),
            "precision_at_k": primary.get("precision_at_k"),
            "mrr": primary.get("mrr"),
            "retrieval_count": int(result.get("retrieved_count") or len(retrieved_sources)),
            "relevant_count": int(result.get("relevant_count") or 0),
            "retrieval_latency_s": retrieval_latency,
            "generation_latency_s": generation_latency,
            "total_latency_s": total_latency,
            "wall_latency_s": wall_latency,
            "error": run_error,
            "metrics_by_k": metrics_by_k,
            "llm_judge": llm_judge,
            "trace": trace,
        }

        rows.append(row)

        if verbose:
            status = "OK" if not run_error else "ERREUR"
            print(
                f"    {status} | retrieval={retrieval_executed} | "
                f"latence={total_latency:.2f}s"
            )

    summary = aggregate(rows, k_values)
    return rows, summary


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def aggregate(rows: Sequence[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, Any]:
    valid = [r for r in rows if not r.get("error")]

    judge_fields = ["faithfulness", "answer_relevancy", "context_relevancy", "correctness", "overall_score"]
    judge_summary = {
        field: mean_or_none(r.get("llm_judge", {}).get(field) for r in valid)
        for field in judge_fields
    }
    judge_summary["evaluated_tests"] = sum(1 for r in valid if r.get("llm_judge", {}).get("overall_score") is not None)
    judge_summary["errors"] = sum(1 for r in valid if r.get("llm_judge", {}).get("error"))
    judge_summary["first_error"] = next((r.get("llm_judge", {}).get("error") for r in valid if r.get("llm_judge", {}).get("error")), None)
    judge_summary["latency_avg_s"] = mean_or_none(r.get("llm_judge", {}).get("latency_s") for r in valid)

    summary: Dict[str, Any] = {
        "tests_total": len(rows),
        "tests_success": len(valid),
        "tests_errors": len(rows) - len(valid),
        "retrieval_behavior_accuracy": mean_or_none(
            1.0 if r["retrieval_behavior_ok"] else 0.0
            for r in valid
            if r["retrieval_behavior_ok"] is not None
        ),
        "answer_keyword_score": mean_or_none(
            r["answer_keyword_score"] for r in valid
        ),
        "llm_judge": judge_summary,
        "latency": {
            "retrieval_avg_s": mean_or_none(r["retrieval_latency_s"] for r in valid),
            "generation_avg_s": mean_or_none(r["generation_latency_s"] for r in valid),
            "total_avg_s": mean_or_none(r["total_latency_s"] for r in valid),
            "total_p50_s": percentile(
                [r["total_latency_s"] for r in valid], 0.50
            ),
            "total_p95_s": percentile(
                [r["total_latency_s"] for r in valid], 0.95
            ),
        },
        "by_k": {},
        "by_category": {},
    }

    for k in k_values:
        key = str(k)
        summary["by_k"][key] = {
            "hit_at_k": mean_or_none(
                (r["metrics_by_k"][key]["hit_at_k"] for r in valid)
            ),
            "recall_at_k": mean_or_none(
                (r["metrics_by_k"][key]["recall_at_k"] for r in valid)
            ),
            "precision_at_k": mean_or_none(
                (r["metrics_by_k"][key]["precision_at_k"] for r in valid)
            ),
            "mrr": mean_or_none(
                (r["metrics_by_k"][key]["mrr"] for r in valid)
            ),
        }

    categories = sorted({str(r["category"]) for r in rows})
    for category in categories:
        cat_rows = [r for r in valid if str(r["category"]) == category]
        summary["by_category"][category] = {
            "count": len(cat_rows),
            "retrieval_behavior_accuracy": mean_or_none(
                1.0 if r["retrieval_behavior_ok"] else 0.0
                for r in cat_rows
                if r["retrieval_behavior_ok"] is not None
            ),
            "answer_keyword_score": mean_or_none(
                r["answer_keyword_score"] for r in cat_rows
            ),
            "faithfulness": mean_or_none(r.get("llm_judge", {}).get("faithfulness") for r in cat_rows),
            "answer_relevancy": mean_or_none(r.get("llm_judge", {}).get("answer_relevancy") for r in cat_rows),
            "context_relevancy": mean_or_none(r.get("llm_judge", {}).get("context_relevancy") for r in cat_rows),
            "correctness": mean_or_none(r.get("llm_judge", {}).get("correctness") for r in cat_rows),
            "overall_score": mean_or_none(r.get("llm_judge", {}).get("overall_score") for r in cat_rows),
            "total_avg_s": mean_or_none(
                r["total_latency_s"] for r in cat_rows
            ),
            "hit_at_max_k": mean_or_none(
                r["metrics_by_k"][str(max(k_values))]["hit_at_k"]
                for r in cat_rows
            ),
        }

    return summary


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "id", "position", "category", "session", "question",
        "expected_retrieval", "retrieval_executed", "retrieval_behavior_ok",
        "expected_sources", "retrieved_sources",
        "answer_keyword_score",
        "faithfulness", "answer_relevancy", "context_relevancy",
        "correctness", "overall_score", "judge_latency_s",
        "judge_error", "judge_reason",
        "hit_at_k", "recall_at_k", "precision_at_k", "mrr",
        "retrieval_count", "relevant_count",
        "retrieval_latency_s", "generation_latency_s",
        "total_latency_s", "wall_latency_s", "error",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            clean["expected_sources"] = " | ".join(clean.get("expected_sources") or [])
            clean["retrieved_sources"] = " | ".join(clean.get("retrieved_sources") or [])
            judge = clean.get("llm_judge") or {}
            clean.update({
                "faithfulness": judge.get("faithfulness"),
                "answer_relevancy": judge.get("answer_relevancy"),
                "context_relevancy": judge.get("context_relevancy"),
                "correctness": judge.get("correctness"),
                "overall_score": judge.get("overall_score"),
                "judge_latency_s": judge.get("latency_s"),
                "judge_error": judge.get("error"),
                "judge_reason": judge.get("reason"),
            })
            clean.pop("metrics_by_k", None)
            clean.pop("llm_judge", None)
            clean.pop("trace", None)
            writer.writerow({field: clean.get(field) for field in fields})


def print_report(summary: Dict[str, Any], k_values: Sequence[int]) -> None:
    print("\n" + "=" * 72)
    print(" ÉVALUATION RAG — TELNET SMARTCONNECT")
    print("=" * 72)

    print(f"Tests : {summary['tests_total']}")
    print(f"Succès : {summary['tests_success']}")
    print(f"Erreurs : {summary['tests_errors']}")

    rba = summary.get("retrieval_behavior_accuracy")
    aks = summary.get("answer_keyword_score")

    print(
        f"Retrieval déclenché correctement : "
        f"{rba * 100:.1f}%"
        if rba is not None
        else "Retrieval déclenché correctement : n/a"
    )
    print(
        f"Score lexical réponses : {aks * 100:.1f}%"
        if aks is not None
        else "Score lexical réponses : n/a"
    )

    judge = summary.get("llm_judge", {})
    print("\nGénération — LLM Judge")
    print("-" * 72)
    for label, key in [("Faithfulness", "faithfulness"), ("Answer Relevancy", "answer_relevancy"), ("Context Relevancy", "context_relevancy"), ("Correctness", "correctness")]:
        value = judge.get(key)
        print(f"{label:<24}: {'n/a' if value is None else f'{value:.3f}'}")
    overall = judge.get("overall_score")
    print(f"{'Note globale':<24}: {'n/a' if overall is None else f'{overall:.2f} / 5'}")
    print(f"{'Tests évalués':<24}: {judge.get('evaluated_tests', 0)}")
    print(f"{'Erreurs du juge':<24}: {judge.get('errors', 0)}")
    if judge.get("errors", 0) and judge.get("first_error"):
        print("Première erreur du juge :")
        print(f"  {judge['first_error']}")


    print("\nRetrieval")
    print("-" * 72)
    print("K     Hit@K     Recall@K   Precision@K   MRR")
    for k in k_values:
        m = summary["by_k"][str(k)]
        def pct(x):
            return "n/a" if x is None else f"{x * 100:7.2f}%"
        def num(x):
            return "n/a" if x is None else f"{x:7.3f}"
        print(
            f"{k:<5}"
            f"{pct(m['hit_at_k']):<11}"
            f"{pct(m['recall_at_k']):<12}"
            f"{pct(m['precision_at_k']):<15}"
            f"{num(m['mrr'])}"
        )

    lat = summary["latency"]
    print("\nLatence")
    print("-" * 72)
    for label, key in [
        ("Retrieval moyenne", "retrieval_avg_s"),
        ("Génération moyenne", "generation_avg_s"),
        ("Total moyenne", "total_avg_s"),
        ("Total P50", "total_p50_s"),
        ("Total P95", "total_p95_s"),
    ]:
        value = lat.get(key)
        print(f"{label:<24}: {'n/a' if value is None else f'{value:.3f}s'}")

    print("\nPar catégorie")
    print("-" * 72)
    for category, m in summary["by_category"].items():
        rba = m["retrieval_behavior_accuracy"]
        h = m["hit_at_max_k"]
        print(
            f"{category:<22} "
            f"n={m['count']:<3} "
            f"retrieval={('n/a' if rba is None else f'{rba*100:.1f}%'):<8} "
            f"Hit@maxK={('n/a' if h is None else f'{h*100:.1f}%'):<8} "
            f"lat={('n/a' if m['total_avg_s'] is None else f'{m['total_avg_s']:.2f}s')}"
        )

    print("=" * 72)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Évalue automatiquement le RAG TELNET SmartConnect.")
    parser.add_argument(
        "--dataset",
        default="tests_rag_smartconnect_eval.json",
        help="Jeu de tests JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="evaluation_results.json",
        help="Résultats détaillés JSON.",
    )
    parser.add_argument(
        "--output-csv",
        default="evaluation_metrics.csv",
        help="Métriques par question CSV.",
    )
    parser.add_argument(
        "--pipeline-module",
        default="rag_projet_corrige.rag_fixed_final.pipeline",
        help="Module Python contenant RAGPipeline.",
    )
    parser.add_argument(
        "--pipeline-class",
        default="RAGPipeline",
        help="Nom de la classe du pipeline.",
    )
    parser.add_argument(
        "--k",
        default="1,3,5,8",
        help="Valeurs de K, ex: 1,3,5,8.",
    )
    parser.add_argument(
        "--judge-model",
        default="mistral",
        help="Modèle Ollama utilisé comme LLM juge.",
    )
    parser.add_argument(
        "--judge-context-chars",
        type=int,
        default=18000,
        help="Taille maximale du contexte envoyé au juge.",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Désactive le LLM juge.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche la progression de chaque test.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tests = dataset.get("tests", dataset if isinstance(dataset, list) else [])
    if not tests:
        raise ValueError("Aucun test trouvé dans le dataset.")

    k_values = sorted({
        int(x.strip())
        for x in args.k.split(",")
        if x.strip()
    })
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--k doit contenir des entiers positifs.")

    print("Chargement du RAG...")
    pipeline_class = load_pipeline_class(args.pipeline_module, args.pipeline_class)
    pipeline = build_pipeline(pipeline_class)

    index_status = load_or_build_index(pipeline)
    print(f"Index : {index_status}")
    print(f"Tests : {len(tests)}")
    print(f"K évalués : {k_values}")
    judge_llm = None
    if not args.no_llm_judge:
        print(f"Chargement du LLM juge : {args.judge_model}")
        judge_llm = create_judge(args.judge_model)

    print(f"LLM Judge : {args.judge_model if judge_llm is not None else 'désactivé'}")
    print(f"Pipeline : {args.pipeline_module}.{args.pipeline_class}")

    rows, summary = run_evaluation(
        pipeline=pipeline,
        tests=tests,
        k_values=k_values,
        verbose=args.verbose,
        judge_llm=judge_llm,
        judge_context_chars=args.judge_context_chars,
    )

    payload = {
        "project": dataset.get("project", "TELNET Support Bot"),
        "dataset": str(dataset_path),
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": {
            "module": args.pipeline_module,
            "class": args.pipeline_class,
            "config": {
                "embedding_model": "BAAI/bge-m3",
                "llm_model": "mistral",
                "retrieval_type": "hybrid",
                "retrieval_k": 8,
                "retrieval_fetch_k": 20,
                "retrieval_lambda": 0.6,
                "hybrid_k": 10,
                "relevance_threshold": None,
                "similarity_score_threshold": 0.40,
                "max_context_documents": 6,
                "use_history": True,
                "max_history": 10,
                "max_history_chars": 5000,
                "enable_query_rewriting": True,
            },
        },
        "k_values": k_values,
        "llm_judge": {
            "enabled": judge_llm is not None,
            "model": args.judge_model if judge_llm is not None else None,
            "context_max_chars": args.judge_context_chars,
            "metrics": ["faithfulness", "answer_relevancy", "context_relevancy", "correctness", "overall_score_0_5"],
        },
        "summary": summary,
        "tests": rows,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv

    save_json(output_json, payload)
    save_csv(output_csv, rows)

    print_report(summary, k_values)

    print(f"\nRésultats détaillés : {output_json}")
    print(f"Métriques CSV       : {output_csv}")
    print("\nRemarque :")
    print("- Hit/Recall/Precision/MRR utilisent les sources attendues annotées.")
    print("- Le score lexical est un indicateur simple, pas une mesure sémantique.")
    print("- Le LLM Judge fournit Faithfulness, Answer Relevancy, Context Relevancy et Correctness.")
    print("- Correctness s'appuie sur expected_behavior + answer_keywords + contexte (pas de réponse de référence complète dans le dataset).")
    print("- Pour l'évaluation historique, les tests partageant la même session gardent le même historique.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
