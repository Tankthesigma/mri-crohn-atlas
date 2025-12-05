import json
import os
import time
import asyncio
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI

# ============ CONFIGURATION ============
INPUT_FILE = "data/candidate_papers.json"
OUTPUT_FILE = "data/extracted_abstract_data.json"
COST_LIMIT = float('inf')  # Unlimited - cost tracking only for logging
LIMIT = 500  # Process top N papers

# Estimated prices per 1M tokens
PRICING = {
    "deepseek/deepseek-chat": {"input": 0.50, "output": 2.00}, # Approximate
    "default": {"input": 1.00, "output": 3.00}
}

class CostTracker:
    def __init__(self, limit: float):
        self.limit = limit
        self.current_cost = 0.0
        self.lock = asyncio.Lock()
        self.stop_event = asyncio.Event()

    async def add_cost(self, model: str, input_tokens: int, output_tokens: int):
        price = PRICING.get(model, PRICING["default"])
        cost = (input_tokens / 1_000_000 * price["input"]) + (output_tokens / 1_000_000 * price["output"])
        
        async with self.lock:
            self.current_cost += cost
            if self.current_cost >= self.limit:
                print(f"\n$$$ COST LIMIT REACHED: ${self.current_cost:.4f} / ${self.limit:.2f} $$$")
                self.stop_event.set()

tracker = CostTracker(COST_LIMIT)

class AbstractExtractor:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", model: str = "deepseek/deepseek-chat"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def extract_data(self, paper: Dict) -> Dict:
        if tracker.stop_event.is_set():
            return {"status": "skipped_cost_limit"}

        text = f"Title: {paper.get('title')}\nAbstract: {paper.get('abstract')}"
        
        prompt = f"""
        Analyze this Crohn's Disease paper abstract.
        Extract:
        1. Sample Size (N)
        2. Indices (VAI, MAGNIFI-CD)
        3. Correlations (r-values)
        4. Fibrosis definition
        
        Paper:
        {text}
        
        Respond with JSON only:
        {{
            "is_relevant": true/false,
            "sample_size": <int or null>,
            "indices": ["VAI", "MAGNIFI-CD", ...],
            "correlations": ["text", ...],
            "fibrosis_def": "<text>"
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            usage = response.usage
            await tracker.add_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            return {"error": str(e)}

    async def process_papers(self):
        try:
            with open(INPUT_FILE, 'r') as f:
                papers = json.load(f)
        except FileNotFoundError:
            print("Input file not found.")
            return

        # Sort by relevance
        papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        print(f"Loaded {len(papers)} papers. Processing top {LIMIT}...")
        
        tasks = []
        processed_count = 0
        final_results = []
        
        # We process sequentially or in small batches to respect rate limits/cost check
        # For simplicity in this script, sequential with async
        
        for i, paper in enumerate(papers):
            if i < LIMIT and not tracker.stop_event.is_set():
                print(f"Processing ({i+1}/{LIMIT}): {paper.get('title')[:40]}...")
                data = await self.extract_data(paper)
                paper["extraction"] = data
                processed_count += 1
            else:
                paper["extraction"] = {"status": "skipped_limit_or_cost"}
            
            final_results.append(paper)
            
            if tracker.stop_event.is_set():
                break
            
            # Incremental save every 10 papers
            if (i + 1) % 10 == 0:
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(final_results, f, indent=2)
                print(f"Saved progress ({i+1}/{LIMIT}) to {OUTPUT_FILE}")

        # Final Save
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"Saved {len(final_results)} papers to {OUTPUT_FILE}")
        print(f"Total Cost: ${tracker.current_cost:.4f}")

if __name__ == "__main__":
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("API Key missing.")
        exit(1)
        
    extractor = AbstractExtractor(api_key=api_key)
    asyncio.run(extractor.process_papers())
