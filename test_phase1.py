"""Quick Phase 1 smoke test — run from project root."""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from app.research_graph import run_research_pipeline

result = run_research_pipeline("tabular fraud detection transformers")

print(f"\n{'='*60}")
print(f"Papers retrieved : {len(result.papers)}")
print(f"Extractions done : {len(result.extractions)}")
print(f"Claims verified  : {len(result.verified_claims)}")
print(f"Benchmark rows   : {len(result.benchmark_table)}")
print(f"\nGAP SUMMARY:\n{result.gap_summary}")
print(f"\nTOP PAPER: {result.papers[0].title if result.papers else 'none'}")
print(f"CITATION COUNT: {result.papers[0].citation_count if result.papers else 0}")
print('='*60)