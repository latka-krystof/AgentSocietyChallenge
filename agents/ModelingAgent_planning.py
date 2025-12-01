import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.llm import LLMBase
from websocietysimulator.llm.vertex_ai_llm import VertexAILLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanningOnlyModule(PlanningBase):
    """
    Deterministic planning module that enumerates every step of the pipeline.
    The rest of the agent simply executes this plan—no memory or reasoning modules.
    """

    def __call__(self, task_description: Dict[str, Any]) -> List[Dict[str, Any]]:
        user_id = task_description.get("user_id", "")
        item_id = task_description.get("item_id", "")

        self.plan = [
            {
                "order": 1,
                "description": "Inspect persona and long-term biases",
                "action": "fetch_user",
                "tool": "interaction_tool.get_user",
                "inputs": {"user_id": user_id},
            },
            {
                "order": 2,
                "description": "Retrieve business metadata",
                "action": "fetch_business",
                "tool": "interaction_tool.get_item",
                "inputs": {"item_id": item_id},
            },
            {
                "order": 3,
                "description": "Collect recent public sentiment about the business",
                "action": "fetch_item_reviews",
                "tool": "interaction_tool.get_reviews",
                "inputs": {"item_id": item_id, "limit": 10},
            },
            {
                "order": 4,
                "description": "Look up the user’s most recent writing to anchor tone",
                "action": "fetch_user_reviews",
                "tool": "interaction_tool.get_reviews",
                "inputs": {"user_id": user_id, "limit": 5},
            },
            {
                "order": 5,
                "description": "Aggregate deterministic rating from persona + sentiment signals",
                "action": "compute_rating",
            },
            {
                "order": 6,
                "description": "Draft review that references concrete evidence gathered above",
                "action": "compose_review",
            },
        ]
        return self.plan


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "they", "been", "were", "them",
    "when", "what", "your", "will", "about", "into", "also", "very", "just", "their", "really",
    "been", "more", "than", "while", "because", "after", "before", "over", "though", "even",
    "through", "back", "here", "there", "where", "into", "once", "ever", "much", "some", "only",
    "like", "make", "made", "sure", "still",
}


class PlanningOnlySimulationAgent(SimulationAgent):
    """
    Simulation agent that relies purely on the planning module.

    The agent:
      1. Generates a deterministic plan (no LLM reasoning/memory)
      2. Executes each sub-task exactly as prescribed
      3. Uses lightweight heuristics to compute the final rating and review
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = PlanningOnlyModule(llm=self.llm)

    def workflow(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        plan = self.planning(task_description=self.task)
        logger.debug("Generated plan: %s", plan)

        for step in sorted(plan, key=lambda s: s["order"]):
            action = step["action"]

            if action == "fetch_user":
                context["user"] = self.interaction_tool.get_user(
                    user_id=step["inputs"]["user_id"]
                )
            elif action == "fetch_business":
                context["business"] = self.interaction_tool.get_item(
                    item_id=step["inputs"]["item_id"]
                )
            elif action == "fetch_item_reviews":
                reviews = self.interaction_tool.get_reviews(
                    item_id=step["inputs"]["item_id"]
                )
                context["item_reviews"] = reviews[: step["inputs"]["limit"]]
            elif action == "fetch_user_reviews":
                reviews = self.interaction_tool.get_reviews(
                    user_id=step["inputs"]["user_id"]
                )
                context["user_reviews"] = reviews[: step["inputs"]["limit"]]
            elif action == "compute_rating":
                context["analysis"] = self._compute_rating(context)
            elif action == "compose_review":
                context["review_text"] = self._compose_review(context)

        final_rating = context.get("analysis", {}).get("final_rating", 3.0)
        review_text = context.get("review_text", "").strip()

        if not review_text:
            review_text = "Service was acceptable overall, nothing remarkable to report."

        return {
            "stars": max(1.0, min(5.0, float(final_rating))),
            "review": review_text[:512],
        }

    def _compute_rating(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_info = context.get("user", {})
        business = context.get("business", {})
        item_reviews = context.get("item_reviews", [])
        user_reviews = context.get("user_reviews", [])

        user_bias = float(user_info.get("average_stars", 3.5))
        business_avg = float(business.get("stars", math.nan))
        recent_item_avg = self._mean_stars(item_reviews[:5])
        user_recent_avg = self._mean_stars(user_reviews[:3])

        # Weight the different components; fall back to global defaults if missing
        rating_candidates = [
            (business_avg, 0.4),
            (recent_item_avg, 0.35),
            (user_bias, 0.15),
            (user_recent_avg, 0.10),
        ]

        weighted_sum = 0.0
        total_weight = 0.0
        for value, weight in rating_candidates:
            if value is not None and not math.isnan(value):
                weighted_sum += value * weight
                total_weight += weight

        if total_weight == 0:
            final_rating = 3.0
        else:
            final_rating = round(max(1.0, min(5.0, weighted_sum / total_weight)))

        themes = self._extract_themes(item_reviews)

        return {
            "final_rating": final_rating,
            "user_bias": user_bias,
            "business_avg": business_avg if not math.isnan(business_avg) else None,
            "recent_item_avg": recent_item_avg,
            "user_recent_avg": user_recent_avg,
            "themes": themes,
        }

    def _compose_review(self, context: Dict[str, Any]) -> str:
        analysis = context.get("analysis", {})
        user = context.get("user", {})
        business = context.get("business", {})
        item_reviews = context.get("item_reviews", [])
        user_reviews = context.get("user_reviews", [])

        business_name = business.get("name", "this business")
        city = business.get("city")
        category = business.get("categories", "local spot")
        rating = analysis.get("final_rating", 3)
        themes = analysis.get("themes", [])

        persona_line = ""
        if user.get("name") and user.get("average_stars"):
            persona_line = (
                f"As someone who typically hovers around {user['average_stars']:.1f} stars, "
            )

        context_line = f"I stopped by {business_name}"
        if city:
            context_line += f" in {city}"
        context_line += f" for a {category.lower()} fix."

        # Without a reasoning LLM we need fixed phrasing so the review stays coherent.
        # These templates stand in for what the reasoning model would normally write.
        if rating >= 4:
            sentiment_line = "Most of what I experienced lined up with the upbeat chatter I've been seeing."
        elif rating == 3:
            sentiment_line = "It landed somewhere in the middle for me—solid fundamentals, but nothing that moved the needle."
        else:
            sentiment_line = "I ran into enough friction that I'd steer friends elsewhere until things tighten up."

        evidence_line = self._build_evidence_line(item_reviews, user_reviews, themes)

        # Same idea here: deterministic closing sentences mimic the role of a reasoning model.
        closing_line = (
            "I'll keep it on my list for specific cravings."
            if rating >= 4
            else "I'd only return if friends insisted."
            if rating == 3
            else "I'll look for other options next time."
        )

        review_parts = [
            persona_line + context_line,
            sentiment_line,
            evidence_line,
            closing_line,
        ]

        return " ".join(part.strip() for part in review_parts if part).strip()

    @staticmethod
    def _mean_stars(reviews: List[Dict[str, Any]]) -> float:
        stars = [float(review.get("stars")) for review in reviews if review.get("stars") is not None]
        return mean(stars) if stars else None

    @staticmethod
    def _extract_themes(reviews: List[Dict[str, Any]], limit: int = 4) -> List[str]:
        texts = [review.get("text", "") for review in reviews if review.get("text")]
        tokens: List[str] = []
        for text in texts:
            for token in re.findall(r"[a-zA-Z']+", text.lower()):
                if token not in STOPWORDS and len(token) > 3:
                    tokens.append(token)
        most_common = [word for word, _ in Counter(tokens).most_common(limit)]
        return most_common

    @staticmethod
    def _build_evidence_line(
        item_reviews: List[Dict[str, Any]],
        user_reviews: List[Dict[str, Any]],
        themes: List[str],
    ) -> str:
        snippets: List[str] = []

        if themes:
            snippets.append(f"Recent visitors keep mentioning {', '.join(themes[:3])}.")

        if item_reviews:
            first_review = item_reviews[0]
            if first_review.get("text"):
                first_sentence = first_review["text"].split(".")[0].strip()
                if first_sentence:
                    snippets.append(f"One of the latest reviews noted: \"{first_sentence}.\"")

        if user_reviews:
            own_recent = user_reviews[0]
            if own_recent.get("text"):
                snippets.append("My last write-up had a similar tone, so this felt consistent.")

        if not snippets:
            snippets.append("Nothing wildly different from the broader Yelp consensus.")

        return " ".join(snippets)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path so we can import from example module
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    logging.getLogger().setLevel(logging.INFO)

    task_set = "yelp"
    num_tasks = 10
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./planning_only_{task_set}_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    simulator = Simulator(data_dir="./dataset", device="auto", cache=False)
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{task_set}/tasks",
        groundtruth_dir=f"./example/track1/{task_set}/groundtruth",
    )

    simulator.set_llm(
        VertexAILLM(
            project_id=os.getenv("GCP_PROJECT_ID") or "agentsocietychallenge",
            location="us-central1",
            model="gemini-2.5-pro",
            use_vertex_ai_embeddings=False,
            embedding_model="text-embedding-004",
        )
    )

    strategies = [
        ("PlanningOnly", PlanningOnlySimulationAgent),
    ]

    try:
        from example.ModelingAgent_baseline import MySimulationAgent as BaselineSimulationAgent

        strategies.append(("Baseline", BaselineSimulationAgent))
    except Exception as import_error:
        logger.warning("Could not import baseline agent: %s", import_error)

    try:
        from example.ModelingAgent_memory_and_reasoning import (
            MySimulationAgentMemoryAndReasoning,
        )

        strategies.append(
            ("MemoryAndReasoning", MySimulationAgentMemoryAndReasoning)
        )
    except Exception as import_error:
        logger.warning("Could not import comparison agent: %s", import_error)

    for name, agent_cls in strategies:
        print(f"\n{'=' * 60}\nRunning strategy: {name}\n{'=' * 60}")
        simulator.set_agent(agent_cls)

        try:
            outputs = simulator.run_simulation(
                number_of_tasks=num_tasks,
                enable_threading=False,
                max_workers=1,
            )

            outputs_path = results_dir / f"{name}_outputs.json"
            outputs_path.write_text(json.dumps(outputs, indent=4))
            print(f"Outputs saved to {outputs_path}")

            evaluation = simulator.evaluate()
            evaluation_path = results_dir / f"{name}_evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation, indent=4))
            print(f"Evaluation saved to {evaluation_path}")

            metrics = evaluation.get("metrics", {})
            if metrics:
                print(
                    f"   Preference Estimation: {metrics.get('preference_estimation')}\n"
                    f"   Review Generation: {metrics.get('review_generation')}\n"
                    f"   Overall Quality: {metrics.get('overall_quality')}"
                )
        except Exception as run_error:
            err_file = results_dir / f"{name}_error.txt"
            err_file.write_text(f"{run_error}\n")
            print(f"❌ Error running {name}: {run_error}")

    print(f"\n{'='*60}")
    print(f"Comparison complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    # Print comparison summary
    print("\n COMPARISON SUMMARY:")
    print("-" * 60)
    try:
        planning_eval_path = results_dir / "PlanningOnly_evaluation.json"
        if planning_eval_path.exists():
            planning_eval = json.loads(planning_eval_path.read_text())
            
            baseline_eval_path = results_dir / "Baseline_evaluation.json"
            if baseline_eval_path.exists():
                baseline_eval = json.loads(baseline_eval_path.read_text())
                
                if 'metrics' in planning_eval and 'metrics' in baseline_eval:
                    planning_metrics = planning_eval['metrics']
                    baseline_metrics = baseline_eval['metrics']
                    
                    print(f"\nPlanningOnly:")
                    print(f"  Preference Estimation: {planning_metrics.get('preference_estimation', 'N/A')}")
                    print(f"  Review Generation: {planning_metrics.get('review_generation', 'N/A')}")
                    print(f"  Overall Quality: {planning_metrics.get('overall_quality', 'N/A')}")
                    
                    print(f"\nBaseline:")
                    print(f"  Preference Estimation: {baseline_metrics.get('preference_estimation', 'N/A')}")
                    print(f"  Review Generation: {baseline_metrics.get('review_generation', 'N/A')}")
                    print(f"  Overall Quality: {baseline_metrics.get('overall_quality', 'N/A')}")
                    
                    # Calculate improvements
                    print(f"\nPlanningOnly vs Baseline:")
                    for metric in ['preference_estimation', 'review_generation', 'overall_quality']:
                        baseline_val = baseline_metrics.get(metric, 0)
                        planning_val = planning_metrics.get(metric, 0)
                        if baseline_val > 0:
                            improvement = ((planning_val - baseline_val) / baseline_val) * 100
                            sign = "+" if improvement > 0 else ""
                            print(f"  {metric}: {sign}{improvement:.2f}%")
    except Exception as e:
        print(f"Could not generate comparison summary: {e}")


