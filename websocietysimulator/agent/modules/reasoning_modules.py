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
        prompt = '''Think through this step by step before writing your review:

Step 1: Analyze your user profile and review history
- What are your typical rating patterns?
- What aspects do you usually focus on in reviews?
- What is your review style (formal, casual, detailed, brief)?

Step 2: Analyze the business
- What are the key attributes of this business?
- What stands out (positively or negatively)?
- How does this compare to similar businesses you've reviewed?

Step 3: Consider similar reviews
- What did other users highlight?
- Are there common themes or concerns?

Step 4: Determine your rating
- Based on your profile, the business attributes, and your typical rating patterns, what rating would you give?
- Consider: Does this business meet, exceed, or fall short of your expectations?

Step 5: Write your review
- Match your typical review style
- Focus on specific aspects that matter to you
- Be consistent with your rating (positive rating = positive review, negative rating = negative review)

{task_description}

IMPORTANT: Your final output MUST be in exactly this format (no other text before or after):
stars: [rating number: 1.0, 2.0, 3.0, 4.0, or 5.0]
review: [your review text]'''
        
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
        rating_prompt = f'''First, determine what rating you would give. Think step by step based on your profile and the business information.

{task_description}

Output ONLY the rating number (1.0, 2.0, 3.0, 4.0, or 5.0) in this format:
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
        review_prompt = f'''You have decided to give this business a {predicted_stars}-star rating. Now write a review that is consistent with this rating.

{task_description}

IMPORTANT: Your output MUST be in exactly this format (no other text before or after):
stars: {predicted_stars}
review: [your review text]'''
        
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
        
        # Initial generation
        initial_prompt = f'''
{task_description}

IMPORTANT: Your output MUST be in exactly this format (no other text before or after):
stars: [rating number: 1.0, 2.0, 3.0, 4.0, or 5.0]
review: [your review text]'''
        
        messages = [{"role": "user", "content": initial_prompt}]
        initial_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Self-refinement with consistency checking
        refine_prompt = f'''Review the following rating and review for consistency and quality:

Initial Output:
{initial_result}

Task Context:
{task_description}

Check for:
1. Consistency: Does the review sentiment match the rating?
   - 4-5 stars should have positive/enthusiastic review
   - 1-2 stars should have negative/critical review
   - 3 stars should have balanced/mixed review
2. Style consistency: Does the review match the user's typical style?
3. Rating appropriateness: Is the rating consistent with the user's typical patterns?

If there are inconsistencies, provide a revised version. Otherwise, confirm the original is good.

IMPORTANT: Your output MUST be in exactly this format (no other text before or after):
stars: [rating number: 1.0, 2.0, 3.0, 4.0, or 5.0]
review: [review text]'''
        
        messages = [{"role": "user", "content": refine_prompt}]
        refined_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Post-process to ensure correct format
        return self._ensure_format(refined_result)
    
    def _ensure_format(self, result: str) -> str:
        """Ensure the result is in the correct format: stars: X\nreview: Y"""
        import re
        stars_match = re.search(r'stars?[:\s]+([1-5](?:\.0)?)', result, re.IGNORECASE)
        review_match = re.search(r'review[:\s]+(.+?)(?:\n\n|\Z)', result, re.IGNORECASE | re.DOTALL)
        
        if stars_match and review_match:
            stars = stars_match.group(1)
            review = review_match.group(1).strip()
            return f'stars: {stars}\nreview: {review}'
        elif stars_match:
            review_part = result.split('review:', 1)[-1] if 'review:' in result.lower() else result.split('stars:', 1)[-1].split(':', 1)[-1] if ':' in result else ''
            return f'stars: {stars_match.group(1)}\nreview: {review_part.strip()}'
        else:
            return result


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
        
        # Step 2: Apply principles
        apply_prompt = f'''Apply these principles to write your review:

Principles:
{principles}

{task_description}

IMPORTANT: Your output MUST be in exactly this format (no other text before or after):
stars: [rating number: 1.0, 2.0, 3.0, 4.0, or 5.0]
review: [your review text]'''
        
        messages = [{"role": "user", "content": apply_prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Post-process to ensure correct format
        return self._ensure_format(reasoning_result)
    
    def _ensure_format(self, result: str) -> str:
        """Ensure the result is in the correct format: stars: X\nreview: Y"""
        import re
        stars_match = re.search(r'stars?[:\s]+([1-5](?:\.0)?)', result, re.IGNORECASE)
        review_match = re.search(r'review[:\s]+(.+?)(?:\n\n|\Z)', result, re.IGNORECASE | re.DOTALL)
        
        if stars_match and review_match:
            stars = stars_match.group(1)
            review = review_match.group(1).strip()
            return f'stars: {stars}\nreview: {review}'
        elif stars_match:
            review_part = result.split('review:', 1)[-1] if 'review:' in result.lower() else result.split('stars:', 1)[-1].split(':', 1)[-1] if ':' in result else ''
            return f'stars: {stars_match.group(1)}\nreview: {review_part.strip()}'
        else:
            return result


class ReasoningCOTWithReflection(ReasoningBase):
    """
    Chain of Thought reasoning followed by reflection and refinement.
    Combines step-by-step reasoning with self-critique.
    """
    def __call__(self, task_description: str, feedback: str = ''):
        examples, task_description = self.process_task_description(task_description)
        
        # Initial COT reasoning
        cot_prompt = '''Think through this step by step:

Step 1: Analyze your user profile and review history
- What are your typical rating patterns?
- What aspects do you usually focus on in reviews?
- What is your review style?

Step 2: Analyze the business
- What are the key attributes?
- What stands out (positively or negatively)?

Step 3: Determine your rating
- Based on your profile and the business, what rating would you give?

Step 4: Write your review
- Match your typical review style
- Be consistent with your rating

{task_description}

IMPORTANT: Your output MUST be in exactly this format (no other text before or after):
stars: [rating number: 1.0, 2.0, 3.0, 4.0, or 5.0]
review: [your review text]'''
        
        messages = [{"role": "user", "content": cot_prompt}]
        initial_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Reflection
        reflection_prompt = f'''Reflect on your initial response:

Initial Response:
{initial_result}

Task Context:
{task_description}

Questions to consider:
1. Is the rating consistent with your typical patterns?
2. Does the review sentiment match the rating?
3. Does the review style match your historical reviews?

If you identify any issues, provide a revised version. Otherwise, confirm the original is good.

IMPORTANT: Your output MUST be in exactly this format (no other text before or after):
stars: [rating number: 1.0, 2.0, 3.0, 4.0, or 5.0]
review: [review text]'''
        
        messages = [{"role": "user", "content": reflection_prompt}]
        final_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Post-process to ensure correct format
        return self._ensure_format(final_result)
    
    def _ensure_format(self, result: str) -> str:
        """Ensure the result is in the correct format: stars: X\nreview: Y"""
        import re
        stars_match = re.search(r'stars?[:\s]+([1-5](?:\.0)?)', result, re.IGNORECASE)
        review_match = re.search(r'review[:\s]+(.+?)(?:\n\n|\Z)', result, re.IGNORECASE | re.DOTALL)
        
        if stars_match and review_match:
            stars = stars_match.group(1)
            review = review_match.group(1).strip()
            return f'stars: {stars}\nreview: {review}'
        elif stars_match:
            review_part = result.split('review:', 1)[-1] if 'review:' in result.lower() else result.split('stars:', 1)[-1].split(':', 1)[-1] if ':' in result else ''
            return f'stars: {stars_match.group(1)}\nreview: {review_part.strip()}'
        else:
            return result

