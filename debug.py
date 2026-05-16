"""
pipeline_debug.py – Full pipeline test for SHL Recommender.
Runs scraper → embedder → retriever → agent sequentially.
Prints detailed outputs to identify where recall is lost.
"""

import sys
import json
import logging
import subprocess
import os
from pathlib import Path

# Configure logging to see everything
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Step 1: Run scraper (if catalog missing)
# ----------------------------------------------------------------------
def step1_scrape():
    if os.path.exists("data/catalog.json"):
        logger.info("✅ catalog.json already exists. Skipping scraping.")
        with open("data/catalog.json") as f:
            return json.load(f)
    logger.info("Step 1: Running scraper.py to generate catalog...")
    result = subprocess.run([sys.executable, "scraper.py"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Scraper failed:\n%s", result.stderr)
        sys.exit(1)
    with open("data/catalog.json") as f:
        return json.load(f)

# ----------------------------------------------------------------------
# Step 2: Run embedder (if index missing)
# ----------------------------------------------------------------------
def step2_embed():
    if os.path.exists("data/faiss_index.bin") and os.path.exists("data/assessments_metadata.pkl"):
        logger.info("✅ FAISS index already exists. Skipping embedding.")
        return
    logger.info("Step 2: Running embedder.py to build FAISS index...")
    result = subprocess.run([sys.executable, "embedder.py"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Embedder failed:\n%s", result.stderr)
        sys.exit(1)
    logger.info("Embedding completed.")

# ----------------------------------------------------------------------
# Step 3: Test retriever directly
# ----------------------------------------------------------------------
def step3_test_retriever(query, test_type=None, tags=None):
    from retriever import SHLRetriever
    logger.info("Step 3: Testing retriever alone")
    retriever = SHLRetriever(top_k_semantic=50, top_k_final=20, use_cross_encoder=False)
    # Retrieve without filters first
    results_no_filter = retriever.retrieve(query=query, test_type=None, tags=None)
    logger.info("Top 20 retrieved names (no filters):")
    for i, r in enumerate(results_no_filter[:20], 1):
        logger.info(f"  {i}. {r['name']} (type: {r['test_type']}, score: {r.get('relevance_score',0):.4f})")
    
    # With filters if provided
    if test_type or tags:
        results_filtered = retriever.retrieve(query=query, test_type=test_type, tags=tags)
        logger.info(f"With filters (test_type={test_type}, tags={tags}) → {len(results_filtered)} results:")
        for i, r in enumerate(results_filtered[:10], 1):
            logger.info(f"  {i}. {r['name']} (type: {r['test_type']})")
    else:
        results_filtered = results_no_filter
    return results_no_filter, results_filtered

# ----------------------------------------------------------------------
# Step 4: Test agent
# ----------------------------------------------------------------------
def step4_test_agent(messages):
    from agent import run_agent
    from retriever import SHLRetriever
    retriever = SHLRetriever(top_k_semantic=50, top_k_final=20, use_cross_encoder=False)
    logger.info("Step 4: Testing agent with conversation:")
    for msg in messages:
        logger.info(f"  {msg['role']}: {msg['content'][:80]}")
    response = run_agent(messages, retriever=retriever)
    logger.info("Agent response:")
    logger.info(f"  reply: {response.reply[:200]}...")
    logger.info(f"  recommendations: {len(response.recommendations)} items")
    for rec in response.recommendations:
        logger.info(f"    - {rec.name} (type: {rec.test_type})")
    logger.info(f"  end_of_conversation: {response.end_of_conversation}")
    return response

# ----------------------------------------------------------------------
# Main test
# ----------------------------------------------------------------------
def main():
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    # Step 1 & 2: Build catalog and index if needed
    catalog = step1_scrape()
    step2_embed()
    
    # Test query that previously failed (Java + stakeholder)
    query = "mid-level Java developer who works closely with stakeholders and needs strong communication skills"
    test_type = None   # we'll test without filter first
    tags = None
    
    # Step 3: Retriever alone
    results_no_filter, results_filtered = step3_test_retriever(query, test_type, tags)
    
    # Check if personality tests like OPQ32r appear in top 20
    personality_found = any("OPQ" in r["name"] or "Motivation" in r["name"] for r in results_no_filter[:20])
    knowledge_found = any("Java" in r["name"] or "Programming" in r["name"] for r in results_no_filter[:20])
    logger.info(f"Personality test in top 20? {personality_found}")
    logger.info(f"Java knowledge test in top 20? {knowledge_found}")
    
    # Step 4: Agent
    messages = [{"role": "user", "content": query}]
    response = step4_test_agent(messages)
    
    # Final check: does agent include both types?
    agent_has_knowledge = any("Java" in rec.name or "Programming" in rec.name for rec in response.recommendations)
    agent_has_personality = any("OPQ" in rec.name or "Motivation" in rec.name for rec in response.recommendations)
    logger.info(f"Agent final recommendations include knowledge test? {agent_has_knowledge}")
    logger.info(f"Agent final recommendations include personality test? {agent_has_personality}")
    
    if not (agent_has_knowledge and agent_has_personality):
        logger.warning("RECALL PROBLEM: Agent missing either Java knowledge or personality test.")
        if not knowledge_found:
            logger.warning("Root cause: Retriever's top 20 did not contain any Java knowledge test.")
        elif not personality_found:
            logger.warning("Root cause: Retriever's top 20 did not contain any personality test.")
        else:
            logger.warning("Root cause: Both tests were in top 20 but agent (LLM reranker or fuzzy lookup) dropped them.")
    else:
        logger.info("SUCCESS: Agent returned both Java knowledge and personality tests.")

if __name__ == "__main__":
    main()