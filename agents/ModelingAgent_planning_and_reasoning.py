import json
import logging
import math
import os
import re
from collections import Counter
from statistics import mean
from typing import Any, Dict, List

from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOTWithReflection
from websocietysimulator.llm import LLMBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import shared planning module and helpers from planning-only agent
# Use relative import to avoid path issues
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from example.ModelingAgent_planning import PlanningOnlyModule, STOPWORDS


class PlanningAndReasoningSimulationAgent(SimulationAgent):
    """
    Simulation agent that combines planning with reasoning.
    
    The agent:
      1. Generates a deterministic plan (same as PlanningOnly)
      2. Executes each sub-task exactly as prescribed
      3. Uses lightweight heuristics to compute the final rating
      4. Uses a reasoning module (ReasoningCOTWithReflection) to generate the review text
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = PlanningOnlyModule(llm=self.llm)
        self.reasoning = ReasoningCOTWithReflection(profile_type_prompt='', memory=None, llm=self.llm)

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
                context["review_text"] = self._compose_review_with_reasoning(context)

        final_rating = context.get("analysis", {}).get("final_rating", 3.0)
        review_text = context.get("review_text", "").strip()

        if not review_text:
            review_text = "Service was acceptable overall, nothing remarkable to report."

        return {
            "stars": max(1.0, min(5.0, float(final_rating))),
            "review": review_text[:512],
        }

    def _compute_rating(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Same rating computation as PlanningOnly agent."""
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

    def _compose_review_with_reasoning(self, context: Dict[str, Any]) -> str:
        """Use reasoning module to generate review text based on gathered context."""
        analysis = context.get("analysis", {})
        user = context.get("user", {})
        business = context.get("business", {})
        item_reviews = context.get("item_reviews", [])
        user_reviews = context.get("user_reviews", [])
        
        final_rating = analysis.get("final_rating", 3.0)
        themes = analysis.get("themes", [])
        
        # Build context for reasoning module
        user_str = str(user)
        business_str = str(business)
        
        # Format similar reviews for context
        similar_reviews = ""
        if item_reviews:
            similar_reviews = "\n".join([
                f"Review: {r.get('text', '')[:200]}" 
                for r in item_reviews[:3]
            ])
        
        task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user_str}

            You need to write a review for this business: {business_str}

            Based on my analysis, I've determined this business deserves a {final_rating}-star rating based on:
            - Business average rating: {analysis.get('business_avg', 'N/A')}
            - Recent visitor sentiment: {analysis.get('recent_item_avg', 'N/A')}
            - My typical rating pattern: {analysis.get('user_bias', 'N/A')}
            - My recent review history: {analysis.get('user_recent_avg', 'N/A')}

            Others have reviewed this business before:
            {similar_reviews}

            Key themes mentioned by other reviewers: {', '.join(themes[:3]) if themes else 'None'}

            Please write a review that:
            1. Is consistent with the {final_rating}-star rating I've determined
            2. Reflects my personal style based on my profile
            3. References specific aspects of the business
            4. Is 2-4 sentences, focusing on personal experience and emotional response
            5. Maintains consistency with my historical review style

            Format your response exactly as follows:
            stars: {final_rating}
            review: [your review]
            '''
        
        try:
            result = self.reasoning(task_description)
            
            # Extract stars and review from reasoning output
            stars_line = [line for line in result.split('\n') if 'stars:' in line.lower()]
            review_line = [line for line in result.split('\n') if 'review:' in line.lower()]
            
            if review_line:
                review_text = review_line[0].split(':', 1)[1].strip()
                return review_text
            else:
                # Fallback: extract review text from result
                if 'review:' in result.lower():
                    review_text = result.split('review:', 1)[-1].strip()
                    return review_text
                else:
                    # Last resort: return the result as-is (minus stars line if present)
                    lines = [l for l in result.split('\n') if 'stars:' not in l.lower()]
                    return ' '.join(lines).strip()
        except Exception as e:
            logger.error(f"Error in reasoning module: {e}")
            # Fallback to deterministic review if reasoning fails
            return self._compose_review_deterministic_fallback(context)

    def _compose_review_deterministic_fallback(self, context: Dict[str, Any]) -> str:
        """Fallback deterministic review generation if reasoning fails."""
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

        if rating >= 4:
            sentiment_line = "Most of what I experienced lined up with the upbeat chatter I've been seeing."
        elif rating == 3:
            sentiment_line = "It landed somewhere in the middle for me—solid fundamentals, but nothing that moved the needle."
        else:
            sentiment_line = "I ran into enough friction that I'd steer friends elsewhere until things tighten up."

        # Import the helper from planning-only agent
        from example.ModelingAgent_planning import PlanningOnlySimulationAgent
        evidence_line = PlanningOnlySimulationAgent._build_evidence_line(item_reviews, user_reviews, themes)

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
        """Same helper as PlanningOnly agent."""
        stars = [float(review.get("stars")) for review in reviews if review.get("stars") is not None]
        return mean(stars) if stars else None

    @staticmethod
    def _extract_themes(reviews: List[Dict[str, Any]], limit: int = 4) -> List[str]:
        """Same helper as PlanningOnly agent."""
        texts = [review.get("text", "") for review in reviews if review.get("text")]
        tokens: List[str] = []
        for text in texts:
            for token in re.findall(r"[a-zA-Z']+", text.lower()):
                if token not in STOPWORDS and len(token) > 3:
                    tokens.append(token)
        most_common = [word for word, _ in Counter(tokens).most_common(limit)]
        return most_common


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datetime import datetime
    
    from websocietysimulator import Simulator
    from websocietysimulator.llm.vertex_ai_llm import VertexAILLM
    
    # Add parent directory to path so we can import from example module
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    logging.getLogger().setLevel(logging.INFO)

    task_set = "yelp"
    num_tasks = 10
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./planning_and_reasoning_{task_set}_{timestamp}")
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
        ("PlanningAndReasoning", PlanningAndReasoningSimulationAgent),
    ]

    try:
        from example.ModelingAgent_planning import PlanningOnlySimulationAgent
        strategies.append(("PlanningOnly", PlanningOnlySimulationAgent))
    except Exception as import_error:
        logger.warning("Could not import planning-only agent: %s", import_error)

    try:
        from example.ModelingAgent_baseline import MySimulationAgent as BaselineSimulationAgent
        strategies.append(("Baseline", BaselineSimulationAgent))
    except Exception as import_error:
        logger.warning("Could not import baseline agent: %s", import_error)

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
            print(f" Error running {name}: {run_error}")

    print(f"\n{'='*60}")
    print(f"Comparison complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    # Print comparison summary
    print("\n COMPARISON SUMMARY:")
    print("-" * 60)
    try:
        all_metrics = {}
        
        # Load all available evaluation results
        for strategy_name in ["PlanningAndReasoning", "PlanningOnly", "Baseline"]:
            eval_path = results_dir / f"{strategy_name}_evaluation.json"
            if eval_path.exists():
                eval_data = json.loads(eval_path.read_text())
                if 'metrics' in eval_data:
                    all_metrics[strategy_name] = eval_data['metrics']
        
        # Print metrics for each strategy
        for strategy_name, metrics in all_metrics.items():
            print(f"\n{strategy_name}:")
            print(f"  Preference Estimation: {metrics.get('preference_estimation', 'N/A')}")
            print(f"  Review Generation: {metrics.get('review_generation', 'N/A')}")
            print(f"  Overall Quality: {metrics.get('overall_quality', 'N/A')}")
        
        # Calculate improvements vs Baseline if available
        if 'Baseline' in all_metrics:
            baseline_metrics = all_metrics['Baseline']
            print(f"\nImprovements vs Baseline:")
            for strategy_name in ["PlanningAndReasoning", "PlanningOnly"]:
                if strategy_name in all_metrics:
                    strategy_metrics = all_metrics[strategy_name]
                    print(f"\n  {strategy_name}:")
                    for metric in ['preference_estimation', 'review_generation', 'overall_quality']:
                        baseline_val = baseline_metrics.get(metric, 0)
                        strategy_val = strategy_metrics.get(metric, 0)
                        if baseline_val > 0:
                            improvement = ((strategy_val - baseline_val) / baseline_val) * 100
                            sign = "+" if improvement > 0 else ""
                            print(f"    {metric}: {sign}{improvement:.2f}%")
        
        # Compare PlanningAndReasoning vs PlanningOnly
        if 'PlanningOnly' in all_metrics and 'PlanningAndReasoning' in all_metrics:
            planning_only_metrics = all_metrics['PlanningOnly']
            planning_reasoning_metrics = all_metrics['PlanningAndReasoning']
            print(f"\nPlanningAndReasoning vs PlanningOnly:")
            for metric in ['preference_estimation', 'review_generation', 'overall_quality']:
                planning_only_val = planning_only_metrics.get(metric, 0)
                planning_reasoning_val = planning_reasoning_metrics.get(metric, 0)
                if planning_only_val > 0:
                    improvement = ((planning_reasoning_val - planning_only_val) / planning_only_val) * 100
                    sign = "+" if improvement > 0 else ""
                    print(f"  {metric}: {sign}{improvement:.2f}%")
    except Exception as e:
        print(f"Could not generate comparison summary: {e}")

