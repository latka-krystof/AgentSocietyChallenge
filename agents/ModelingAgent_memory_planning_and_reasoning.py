import json
import logging
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOTWithReflection
from websocietysimulator.agent.modules.memory_modules import MemoryUserProfile
from websocietysimulator.llm import LLMBase
from websocietysimulator.llm.vertex_ai_llm import VertexAILLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import shared planning module and helpers from planning-only agent
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from agents.ModelingAgent_planning import PlanningOnlyModule, STOPWORDS


class MemoryPlanningAndReasoningSimulationAgent(SimulationAgent):
    """
    Simulation agent that combines memory, planning, and reasoning.
    
    The agent:
      1. Uses PlanningOnlyModule for deterministic plan execution
      2. Uses MemoryUserProfile to store and retrieve relevant reviews
      3. Uses PlanningOnlyModule's heuristics to compute the final rating
      4. Uses ReasoningCOTWithReflection to generate the review text with memory context
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = PlanningOnlyModule(llm=self.llm)
        self.memory = MemoryUserProfile(llm=self.llm)
        self.reasoning = ReasoningCOTWithReflection(profile_type_prompt='', memory=None, llm=self.llm)

    def workflow(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        plan = self.planning(task_description=self.task)
        logger.debug("Generated plan: %s", plan)

        # Execute plan steps
        # Note: We skip "compute_rating" step since we use LLM-based rating computation
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
                # Store item reviews in memory
                for review in context["item_reviews"]:
                    review_text = review.get("text", "")
                    if review_text:
                        self.memory(f'review: {review_text}')
            elif action == "fetch_user_reviews":
                reviews = self.interaction_tool.get_reviews(
                    user_id=step["inputs"]["user_id"]
                )
                context["user_reviews"] = reviews[: step["inputs"]["limit"]]
            elif action == "compose_review":
                # Retrieve similar reviews from memory before composing
                review_similar = ""
                if context.get("user_reviews"):
                    user_review_text = context["user_reviews"][0].get("text", "")
                    if user_review_text:
                        review_similar = self.memory(user_review_text)
                context["review_similar"] = review_similar
                # Use reasoning module to determine both rating and review
                result = self._compose_review_with_reasoning(context)
                context["rating_and_review"] = result

        # Extract rating and review from reasoning output
        rating_and_review = context.get("rating_and_review", {})
        final_rating = rating_and_review.get("stars", 3.0)
        review_text = rating_and_review.get("review", "").strip()

        if not review_text:
            review_text = "Service was acceptable overall, nothing remarkable to report."

        return {
            "stars": max(1.0, min(5.0, float(final_rating))),
            "review": review_text[:512],
        }

    def _compose_review_with_reasoning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use reasoning module to determine rating and generate review text with memory context."""
        user = context.get("user", {})
        business = context.get("business", {})
        item_reviews = context.get("item_reviews", [])
        user_reviews = context.get("user_reviews", [])
        review_similar = context.get("review_similar", "")
        
        # Build context for reasoning module
        user_str = str(user)
        business_str = str(business)
        
        # Compute some helpful statistics for the LLM
        user_bias = float(user.get("average_stars", 3.5)) if user.get("average_stars") else None
        business_avg = float(business.get("stars")) if business.get("stars") else None
        recent_item_avg = self._mean_stars(item_reviews[:5])
        user_recent_avg = self._mean_stars(user_reviews[:3])
        themes = self._extract_themes(item_reviews)
        
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

            Context information:
            - Business average rating: {business_avg if business_avg is not None else 'N/A'}
            - Recent visitor sentiment (average of recent reviews): {recent_item_avg if recent_item_avg is not None else 'N/A'}
            - My typical rating pattern: {user_bias if user_bias is not None else 'N/A'}
            - My recent review history (average): {user_recent_avg if user_recent_avg is not None else 'N/A'}

            Others have reviewed this business before:
            {similar_reviews}

            Key themes mentioned by other reviewers: {', '.join(themes[:3]) if themes else 'None'}

            Based on my memory of similar reviews I've seen, here's what stands out: {review_similar if review_similar else 'No similar reviews found in memory.'}

            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:
            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards
            - Incorporate insights from your memory of similar reviews

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            '''
        
        try:
            result = self.reasoning(task_description)
            
            # Extract stars and review from reasoning output
            stars_line = [line for line in result.split('\n') if 'stars:' in line.lower()]
            review_line = [line for line in result.split('\n') if 'review:' in line.lower()]
            
            if stars_line and review_line:
                try:
                    stars = float(stars_line[0].split(':')[1].strip())
                    review_text = review_line[0].split(':', 1)[1].strip()
                    return {"stars": stars, "review": review_text}
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing stars/review from reasoning output: {e}")
            elif review_line:
                # If we have review but no stars, try to extract rating from review or use default
                review_text = review_line[0].split(':', 1)[1].strip()
                # Try to find rating in the result
                if 'stars:' in result.lower():
                    try:
                        stars = float([l for l in result.split('\n') if 'stars:' in l.lower()][0].split(':')[1].strip())
                    except:
                        stars = 3.0
                else:
                    stars = 3.0
                return {"stars": stars, "review": review_text}
            else:
                # Fallback: extract review text from result
                if 'review:' in result.lower():
                    review_text = result.split('review:', 1)[-1].strip()
                    # Try to extract stars
                    if 'stars:' in result.lower():
                        try:
                            stars = float([l for l in result.split('\n') if 'stars:' in l.lower()][0].split(':')[1].strip())
                        except:
                            stars = 3.0
                    else:
                        stars = 3.0
                    return {"stars": stars, "review": review_text}
                else:
                    # Last resort: return the result as review with default rating
                    lines = [l for l in result.split('\n') if 'stars:' not in l.lower()]
                    review_text = ' '.join(lines).strip()
                    return {"stars": 3.0, "review": review_text}
        except Exception as e:
            logger.error(f"Error in reasoning module: {e}")
            # Fallback to deterministic review if reasoning fails
            return self._compose_review_deterministic_fallback(context)

    def _compose_review_deterministic_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback deterministic review generation if reasoning fails."""
        user = context.get("user", {})
        business = context.get("business", {})
        item_reviews = context.get("item_reviews", [])
        user_reviews = context.get("user_reviews", [])

        # Compute a simple deterministic rating as fallback
        user_bias = float(user.get("average_stars", 3.5)) if user.get("average_stars") else 3.5
        business_avg = float(business.get("stars", 3.5)) if business.get("stars") else 3.5
        recent_item_avg = self._mean_stars(item_reviews[:5]) or 3.5
        
        # Simple average as fallback
        rating = round(max(1.0, min(5.0, (user_bias + business_avg + recent_item_avg) / 3.0)))

        business_name = business.get("name", "this business")
        city = business.get("city")
        category = business.get("categories", "local spot")
        themes = self._extract_themes(item_reviews)

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
        from agents.ModelingAgent_planning import PlanningOnlySimulationAgent
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

        review_text = " ".join(part.strip() for part in review_parts if part).strip()
        return {"stars": float(rating), "review": review_text}

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
    from pathlib import Path
    
    logging.getLogger().setLevel(logging.INFO)

    task_set = "yelp"
    num_tasks = 50
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./memory_planning_reasoning_{task_set}_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    simulator = Simulator(data_dir="./dataset", device="auto", cache=False)
    simulator.set_task_and_groundtruth(
        task_dir=f"./agents/track1/{task_set}/tasks",
        groundtruth_dir=f"./agents/track1/{task_set}/groundtruth",
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

    # Start with the new combined agent
    strategies = [
        ("MemoryPlanningAndReasoning", MemoryPlanningAndReasoningSimulationAgent),
    ]

    # Import all other agents for comparison
    try:
        from agents.ModelingAgent_baseline import MySimulationAgent as BaselineSimulationAgent
        strategies.append(("Baseline", BaselineSimulationAgent))
    except Exception as import_error:
        logger.warning("Could not import baseline agent: %s", import_error)

    try:
        from agents.ModelingAgent_planning import PlanningOnlySimulationAgent
        strategies.append(("PlanningOnly", PlanningOnlySimulationAgent))
    except Exception as import_error:
        logger.warning("Could not import planning-only agent: %s", import_error)

    try:
        from agents.ModelingAgent_planning_and_reasoning import PlanningAndReasoningSimulationAgent
        strategies.append(("PlanningAndReasoning", PlanningAndReasoningSimulationAgent))
    except Exception as import_error:
        logger.warning("Could not import planning+reasoning agent: %s", import_error)

    try:
        from agents.ModelingAgent_memory_and_reasoning import MySimulationAgentMemoryAndReasoning
        strategies.append(("MemoryAndReasoning", MySimulationAgentMemoryAndReasoning))
    except Exception as import_error:
        logger.warning("Could not import memory+reasoning agent: %s", import_error)

    try:
        from agents.ModelingAgent_memory import MySimulationAgentUserProfile
        strategies.append(("Memory", MySimulationAgentUserProfile))
    except Exception as import_error:
        logger.warning("Could not import memory-only agent: %s", import_error)

    try:
        from agents.ModelingAgent_reasoning import MySimulationAgentCOT
        strategies.append(("Reasoning", MySimulationAgentCOT))
    except Exception as import_error:
        logger.warning("Could not import reasoning-only agent: %s", import_error)

    # Run all strategies
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
            logger.error(f" Error running {name}: {run_error}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Comparison complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    # Print comprehensive comparison summary
    print("\n COMPREHENSIVE COMPARISON SUMMARY:")
    print("-" * 60)
    try:
        all_metrics = {}
        
        # Load all available evaluation results
        strategy_names = [
            "MemoryPlanningAndReasoning",
            "Baseline",
            "PlanningOnly",
            "PlanningAndReasoning",
            "MemoryAndReasoning",
            "Memory",
            "Reasoning"
        ]
        
        for strategy_name in strategy_names:
            eval_path = results_dir / f"{strategy_name}_evaluation.json"
            if eval_path.exists():
                eval_data = json.loads(eval_path.read_text())
                if 'metrics' in eval_data:
                    all_metrics[strategy_name] = eval_data['metrics']
        
        # Print metrics for each strategy
        print("\n Metrics by Strategy:")
        for strategy_name in strategy_names:
            if strategy_name in all_metrics:
                metrics = all_metrics[strategy_name]
                print(f"\n{strategy_name}:")
                print(f"  Preference Estimation: {metrics.get('preference_estimation', 'N/A')}")
                print(f"  Review Generation: {metrics.get('review_generation', 'N/A')}")
                print(f"  Overall Quality: {metrics.get('overall_quality', 'N/A')}")
        
        # Calculate improvements vs Baseline if available
        if 'Baseline' in all_metrics:
            baseline_metrics = all_metrics['Baseline']
            print(f"\n Improvements vs Baseline:")
            for strategy_name in ["MemoryPlanningAndReasoning", "PlanningOnly", "PlanningAndReasoning", 
                                 "MemoryAndReasoning", "Memory", "Reasoning"]:
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
        
        # Compare MemoryPlanningAndReasoning vs other combined strategies
        if 'MemoryPlanningAndReasoning' in all_metrics:
            combined_metrics = all_metrics['MemoryPlanningAndReasoning']
            print(f"\n🎯 MemoryPlanningAndReasoning vs Other Combined Strategies:")
            for strategy_name in ["PlanningAndReasoning", "MemoryAndReasoning"]:
                if strategy_name in all_metrics:
                    other_metrics = all_metrics[strategy_name]
                    print(f"\n  vs {strategy_name}:")
                    for metric in ['preference_estimation', 'review_generation', 'overall_quality']:
                        other_val = other_metrics.get(metric, 0)
                        combined_val = combined_metrics.get(metric, 0)
                        if other_val > 0:
                            improvement = ((combined_val - other_val) / other_val) * 100
                            sign = "+" if improvement > 0 else ""
                            print(f"    {metric}: {sign}{improvement:.2f}%")
        
    except Exception as e:
        logger.error(f"Could not generate comparison summary: {e}")
        import traceback
        traceback.print_exc()

