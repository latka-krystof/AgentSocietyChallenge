from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
from websocietysimulator.llm import LLMBase
from websocietysimulator.llm.vertex_ai_llm import VertexAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningBase,
    ReasoningCOTSimulation,
    ReasoningMultiStepSimulation,
    ReasoningSelfRefineSimulation,
    ReasoningStepBackSimulation,
    ReasoningCOTWithReflection
)
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
import logging
logging.basicConfig(level=logging.INFO)

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None', 
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class MySimulationAgentCOT(SimulationAgent):
    """Participant's implementation of SimulationAgent with Chain of Thought reasoning."""
    
    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgentCOT"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningCOTSimulation(profile_type_prompt='', memory=None, llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in sub_task['description']:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

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
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            '''
            result = self.reasoning(task_description)
            
            try:
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error:', result)

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]
                
            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            import traceback
            print(f"Error in workflow: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            return {
                "stars": 0,
                "review": ""
            }


class MySimulationAgentMultiStep(SimulationAgent):
    """Participant's implementation of SimulationAgent with Multi-Step reasoning."""
    
    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgentMultiStep"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningMultiStepSimulation(profile_type_prompt='', memory=None, llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in sub_task['description']:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

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
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            '''
            result = self.reasoning(task_description)
            
            try:
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error:', result)

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]
                
            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            import traceback
            print(f"Error in workflow: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            return {
                "stars": 0,
                "review": ""
            }


class MySimulationAgentSelfRefine(SimulationAgent):
    """Participant's implementation of SimulationAgent with Self-Refinement reasoning."""
    
    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgentSelfRefine"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningSelfRefineSimulation(profile_type_prompt='', memory=None, llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in sub_task['description']:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

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
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            '''
            result = self.reasoning(task_description)
            
            try:
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error:', result)

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]
                
            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            import traceback
            print(f"Error in workflow: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            return {
                "stars": 0,
                "review": ""
            }


class MySimulationAgentCOTWithReflection(SimulationAgent):
    """Participant's implementation of SimulationAgent with Chain of Thought + Reflection reasoning."""
    
    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgentCOTWithReflection"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningCOTWithReflection(profile_type_prompt='', memory=None, llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in sub_task['description']:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

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
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            '''
            result = self.reasoning(task_description)
            
            try:
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error:', result)

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]
                
            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            import traceback
            print(f"Error in workflow: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            return {
                "stars": 0,
                "review": ""
            }


if __name__ == "__main__":
    import os
    from datetime import datetime
    
    # Set the data
    task_set = "yelp"  # "goodreads" or "amazon" or "yelp"
    num_tasks = 10  # Number of tasks to run for each strategy
    
    # Create results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./reasoning_comparison_{task_set}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}\n")
    
    # Initialize simulator once
    simulator = Simulator(data_dir="./dataset", device="auto", cache=False)
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{task_set}/tasks", 
        groundtruth_dir=f"./example/track1/{task_set}/groundtruth"
    )
    
    # Set LLM once (shared across all strategies)
    simulator.set_llm(VertexAILLM(
        project_id="agentsocietychallenge",
        location="us-central1",
        model="gemini-2.5-pro",  # Available: gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash-001, etc.
        use_vertex_ai_embeddings=False,  # Set to True to use Vertex AI embeddings (text-embedding-004, etc.)
        embedding_model="text-embedding-004"  # Only used if use_vertex_ai_embeddings=True
        # Note: Default uses sentence-transformers (fast, free, local)
    ))
    
    # Import baseline agent
    from ModelingAgent_baseline import MySimulationAgent as BaselineAgent
    
    # Define all strategies to test
    strategies = [
        ("Baseline", BaselineAgent),
        ("COT", MySimulationAgentCOT),
        ("MultiStep", MySimulationAgentMultiStep),
        ("SelfRefine", MySimulationAgentSelfRefine),
        ("COTWithReflection", MySimulationAgentCOTWithReflection),
    ]
    
    # Loop through all strategies
    for strategy_name, agent_class in strategies:
        print(f"\n{'='*60}")
        print(f"Running strategy: {strategy_name}")
        print(f"{'='*60}")
        
        try:
            # Set the agent
            simulator.set_agent(agent_class)
            
            # Run the simulation
            # NOTE: Threading disabled due to ChromaDB thread safety issues with concurrent initialization
            outputs = simulator.run_simulation(
                number_of_tasks=num_tasks, 
                enable_threading=False, 
                max_workers=1
            )
            
            # Save simulation outputs
            outputs_file = os.path.join(results_dir, f"{strategy_name}_outputs.json")
            with open(outputs_file, 'w') as f:
                json.dump(outputs, f, indent=4)
            print(f"✅ Outputs saved to: {outputs_file}")
            print(f"   Generated {len(outputs)} reviews")
            
            # Evaluate the agent
            evaluation_results = simulator.evaluate()
            evaluation_file = os.path.join(results_dir, f"{strategy_name}_evaluation.json")
            with open(evaluation_file, 'w') as f:
                json.dump(evaluation_results, f, indent=4)
            print(f"✅ Evaluation saved to: {evaluation_file}")
            
        except Exception as e:
            print(f"❌ Error running {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            # Save error info
            error_file = os.path.join(results_dir, f"{strategy_name}_error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error running {strategy_name}:\n")
                f.write(f"{str(e)}\n\n")
                f.write(traceback.format_exc())
    
    print(f"\n{'='*60}")
    print(f"✅ All strategies completed!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")

