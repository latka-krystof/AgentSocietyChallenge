from collections import Counter
import re

class ReasoningBase:
    def __init__(self, profile_type_prompt, memory, llm):
        """
        Initialize the reasoning base class
        
        Args:
            profile_type_prompt: Profile type prompt
            memory: Memory module
            llm: LLM instance used to generate reasoning
        """
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm = llm
    
    def process_task_description(self, task_description):
        examples = ''
        return examples, task_description

class ReasoningIO(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        
        return reasoning_result
    
class ReasoningCOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result

class ReasoningCOTSC(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.1,
            n=5
        )
        string_counts = Counter(reasoning_results)
        reasoning_result = string_counts.most_common(1)[0][0]
        return reasoning_result
    
class ReasoningTOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.1,
            n=3
        )
        reasoning_result = self.get_votes(task_description, reasoning_results, examples)
        return reasoning_result
    def get_votes(self, task_description, reasoning_results, examples):
        if 'think'  in reasoning_results[0].lower():
            return reasoning_results[0]
        prompt = '''Given the reasoning process for two completed tasks and one ongoing task, and several answers for the next step, decide which answer best follows the reasoning process for example command format. Output "The best answer is {{s}}", where s is the integer id chosen.
Here are some examples.
{examples}
Here is the task:
{task_description}

'''     
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        for i, y in enumerate(reasoning_results, 1):
            prompt += f'Answer {i}:\n{y}\n'
        vote_outputs = self.llm(
            messages=messages,
            temperature=0.7,
            n=5
        )
        vote_results = [0] * len(reasoning_results)
        for vote_output in vote_outputs:
            pattern = r".*best answer is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(len(reasoning_results)):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        ids = list(range(len(reasoning_results)))
        select_id = sorted(ids, key=lambda x: vote_results[x], reverse=True)[0]
        return reasoning_results[select_id]

class ReasoningDILU(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        messages = [
            {
                "role": "system",
                "content": '''You are ChatGPT, a large language model trained by OpenAI. Now you act as a real human user on Yelp. You will be given a detailed description of the scenario of current frame along with your history of previous decisions. 
'''
            },
            {
                "role": "user",
                "content": f'''Above messages are some examples of how you make a step successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a step for the current scenario. Your instructions must follow the examples.
Here are two examples.
{examples}
Here is the task:
{task_description}'''
            }
        ]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result

class ReasoningSelfRefine(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        reasoning_result = self.refine(reasoning_result)
        return reasoning_result
    def refine(self, reasoning_result):
        prompt = f'''Reflect on the reasoning process and identify any potential errors or areas for improvement. Provide a revised version of the reasoning if necessary.
Here is the original reasoning:
{reasoning_result}
'''     
        messages = [{"role": "user", "content": prompt}]
        feedback_result = self.llm(
            messages=messages,
            temperature=0.0
        )
        return feedback_result
        
class ReasoningStepBack(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        self.principle = self.stepback(task_description)
            
        prompt = f'''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}{self.principle}
Here is the task:
{task_description}'''
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result
    def stepback(self, task_description):
        stepback_prompt = f'''What common sense, instruction structure is involved in solving this task?
{task_description}'''
        messages = [{"role": "user", "content": stepback_prompt}]
        principle = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return principle


# Specialized reasoning modules for simulation tasks

class ReasoningCOTSimulation(ReasoningBase):
    """
    Chain of Thought reasoning specifically designed for simulation tasks.
    Provides structured step-by-step reasoning for rating and review generation.
    """
    def __call__(self, task_description: str, feedback: str = ''):
        examples, task_description = self.process_task_description(task_description)
        # Minimal guidance - let the task_description do the heavy lifting
        prompt = '''Think step by step: First consider your user profile and typical rating patterns, then analyze the business, and finally write a review that matches your style and rating.

{task_description}'''
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        return reasoning_result

class ReasoningMultiStepSimulation(ReasoningBase):
    """
    Multi-step reasoning that first predicts rating, then generates review based on that rating.
    Ensures consistency between rating and review sentiment.
    """
    def __call__(self, task_description: str, feedback: str = ''):
        examples, task_description = self.process_task_description(task_description)
        
        # Step 1: Predict rating first
        rating_prompt = f'''Based on the information below, determine what rating you would give (1.0, 2.0, 3.0, 4.0, or 5.0).

{task_description}

Output ONLY in this format:
stars: [rating]'''
        
        messages = [{"role": "user", "content": rating_prompt}]
        rating_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=200
        )
        
        # Extract rating
        try:
            stars_line = [line for line in rating_result.split('\n') if 'stars:' in line.lower()][0]
            predicted_stars = float(stars_line.split(':')[1].strip())
        except:
            # Fallback: try to extract any number
            numbers = re.findall(r'\d+\.?\d*', rating_result)
            predicted_stars = float(numbers[0]) if numbers else 3.0
            predicted_stars = max(1.0, min(5.0, predicted_stars))  # Clamp to 1-5
        
        # Step 2: Generate review based on predicted rating
        review_prompt = f'''You have decided to give this business a {predicted_stars}-star rating. Write a review that is consistent with this {predicted_stars}-star rating.

{task_description}'''
        
        messages = [{"role": "user", "content": review_prompt}]
        review_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Ensure format is correct
        return self._ensure_format(review_result, predicted_stars)
    
    def _ensure_format(self, result: str, default_stars: float = None) -> str:
        """Ensure the result is in the correct format: stars: X\nreview: Y"""
        import re
        # Check if format is already correct
        if 'stars:' in result.lower() and 'review:' in result.lower():
            # Try to extract and reformat to ensure consistency
            stars_match = re.search(r'stars?[:\s]+([1-5](?:\.0)?)', result, re.IGNORECASE)
            review_match = re.search(r'review[:\s]+(.+?)(?:\n\n|\Z)', result, re.IGNORECASE | re.DOTALL)
            
            if stars_match and review_match:
                stars = stars_match.group(1)
                review = review_match.group(1).strip()
                return f'stars: {stars}\nreview: {review}'
        
        # If format is missing, add it
        if default_stars is not None:
            # Extract review text (everything after "review:" or everything if no "review:" found)
            if 'review:' in result.lower():
                review_text = result.split('review:', 1)[-1].strip()
            else:
                review_text = result.strip()
            return f'stars: {default_stars}\nreview: {review_text}'
        
        return result


class ReasoningSelfRefineSimulation(ReasoningBase):
    """
    Self-refinement reasoning with consistency checking for simulation tasks.
    Generates initial rating and review, then refines for consistency.
    """
    def __call__(self, task_description: str, feedback: str = ''):
        examples, task_description = self.process_task_description(task_description)
        
        # Initial generation - keep it simple like baseline
        initial_prompt = f'''
{task_description}'''
        
        messages = [{"role": "user", "content": initial_prompt}]
        initial_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Self-refinement with consistency checking - keep it minimal
        refine_prompt = f'''Check if the review sentiment matches the rating. If inconsistent, provide a revised version. Otherwise, keep the original.

Initial Output:
{initial_result}

{task_description}'''
        
        messages = [{"role": "user", "content": refine_prompt}]
        refined_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        return refined_result

class ReasoningStepBackSimulation(ReasoningBase):
    """
    Step-back reasoning for simulation tasks. First identifies principles,
    then applies them to generate rating and review.
    """
    def __call__(self, task_description: str, feedback: str = ''):
        examples, task_description = self.process_task_description(task_description)
        
        # Step 1: Identify principles
        stepback_prompt = f'''What are the key principles for writing an authentic Yelp review as this specific user?

Consider:
- How do users typically rate businesses based on their profile?
- What factors influence rating decisions?
- How should review style match user history?
- What makes a review authentic and consistent?

Task context:
{task_description}'''
        
        messages = [{"role": "user", "content": stepback_prompt}]
        principles = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=300
        )
        
        # Step 2: Apply principles - minimal
        apply_prompt = f'''Apply these principles:

{principles}

{task_description}'''
        
        messages = [{"role": "user", "content": apply_prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        return reasoning_result

class ReasoningCOTWithReflection(ReasoningBase):
    """
    Chain of Thought reasoning followed by reflection and refinement.
    Combines step-by-step reasoning with self-critique.
    """
    def __call__(self, task_description: str, feedback: str = ''):
        examples, task_description = self.process_task_description(task_description)
        
        # Initial COT reasoning - minimal guidance
        cot_prompt = '''Think step by step: consider your profile, analyze the business, determine rating, then write review.

{task_description}'''
        
        messages = [{"role": "user", "content": cot_prompt}]
        initial_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Reflection - minimal and focused
        reflection_prompt = f'''Check if the review sentiment matches the rating and if it's consistent with your profile. If not, revise.

Initial Response:
{initial_result}

{task_description}'''
        
        messages = [{"role": "user", "content": reflection_prompt}]
        final_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )

        return final_result

