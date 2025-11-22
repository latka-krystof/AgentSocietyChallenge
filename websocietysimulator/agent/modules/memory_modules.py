import os
import re
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import shutil
import uuid

class MemoryBase:
    def __init__(self, memory_type: str, llm) -> None:
        """
        Initialize the memory base class
        
        Args:
            memory_type: Type of memory
            llm: LLM instance used to generate memory-related text
        """
        self.llm = llm
        self.embedding = self.llm.get_embedding_model()
        db_path = os.path.join('./db', memory_type, f'{str(uuid.uuid4())}')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )

    def __call__(self, current_situation: str = ''):
        if 'review:' in current_situation:
            self.addMemory(current_situation.replace('review:', ''))
        else:
            return self.retriveMemory(current_situation)

    def retriveMemory(self, query_scenario: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def addMemory(self, current_situation: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

class MemoryDILU(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='dilu', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query scenario
        task_name = query_scenario
        
        # Return empty string if memory is empty
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
            
        # Extract task trajectories from results
        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]
        
        # Join trajectories with newlines and return
        return '\n'.join(task_trajectories)

    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

class MemoryGenerative(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='generative', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Get top 3 similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=3)
            
        fewshot_results = []
        importance_scores = []

        # Score each memory's relevance
        for result in similarity_results:
            trajectory = result[0].metadata['task_trajectory']
            fewshot_results.append(trajectory)
            
            # Generate prompt to evaluate importance
            prompt = f'''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Your output format should be:
Score: '''

            # Get importance score
            response = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1, stop_strs=['\n'])
            score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 0
            importance_scores.append(score)

        # Return trajectory with highest importance score
        max_score_idx = importance_scores.index(max(importance_scores))
        return similarity_results[max_score_idx][0].metadata['task_trajectory']
    
    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

class MemoryTP(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='tp', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from scenario
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
            
        # Generate plans based on similar experiences
        experience_plans = []
        task_description = query_scenario
        
        for result in similarity_results:
            prompt = f"""You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather use the successful case to think about the strategy and path you took to attempt to complete the task in the ongoing task. Devise a concise, new plan of action that accounts for your task with reference to specific actions that you should have taken. You will need this later to solve the task. Give your plan after "Plan".
Success Case:
{result[0].metadata['task_trajectory']}
Ongoing task:
{task_description}
Plan:
"""
            experience_plans.append(self.llm(messaage=prompt, temperature=0.1))
            
        return 'Plan from successful attempt in similar task:\n' + '\n'.join(experience_plans)

    def addMemory(self, current_situation: str):
        # Extract task name
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

class MemoryVoyager(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='voyager', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query
        task_name = query_scenario
        
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(task_name, k=1)
        
        # Extract trajectories from results
        memory_trajectories = [result[0].metadata['task_trajectory'] 
                             for result in similarity_results]
                             
        return '\n'.join(memory_trajectories)

    def addMemory(self, current_situation: str):
        # Prompt template for summarizing trajectory
        voyager_prompt = '''You are a helpful assistant that writes a description of the task resolution trajectory.

        1) Try to summarize the trajectory in no more than 6 sentences.
        2) Your response should be a single line of text.

        For example:

Please fill in this part yourself

        Trajectory:
        '''
        
        # Generate summarized trajectory
        prompt = voyager_prompt + current_situation
        trajectory_summary = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1)
        
        # Create document with metadata
        doc = Document(
            page_content=trajectory_summary,
            metadata={
                "task_description": trajectory_summary,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([doc])


# Advanced Memory Modules for Simulation Tasks

class MemoryTopK(MemoryBase):
    """
    Memory module that retrieves top-K similar reviews instead of just 1.
    Useful for getting more context from similar past experiences.
    """
    def __init__(self, llm, k=3):
        """
        Initialize MemoryTopK
        
        Args:
            llm: LLM instance
            k: Number of top similar memories to retrieve (default: 3)
        """
        super().__init__(memory_type='topk', llm=llm)
        self.k = k

    def retriveMemory(self, query_scenario: str):
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find top-K similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=self.k)
        
        # Extract task trajectories from results
        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]
        
        # Join trajectories with newlines and return
        return '\n'.join(task_trajectories)

    def addMemory(self, current_situation: str):
        # Create document with metadata
        memory_doc = Document(
            page_content=current_situation,
            metadata={
                "task_name": current_situation,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])


class MemoryTemporalWeighted(MemoryBase):
    """
    Memory module that weights reviews by recency.
    More recent reviews are given higher weight in retrieval.
    """
    def __init__(self, llm, k=3, recency_weight=0.3):
        """
        Initialize MemoryTemporalWeighted
        
        Args:
            llm: LLM instance
            k: Number of top memories to retrieve (default: 3)
            recency_weight: Weight for recency factor (0.0-1.0, default: 0.3)
        """
        super().__init__(memory_type='temporal', llm=llm)
        self.k = k
        self.recency_weight = recency_weight
        self.memory_order = []  # Track order of memories (most recent last)

    def retriveMemory(self, query_scenario: str):
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
        
        # Get more candidates than needed for weighting
        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=min(self.k * 2, self.scenario_memory._collection.count()))
        
        # Weight by similarity and recency
        weighted_results = []
        for result_tuple in similarity_results:
            result = result_tuple[0]  # Document object
            score = result_tuple[1]   # Similarity score (lower = more similar)
            trajectory = result.metadata['task_trajectory']
            # Recency: more recent memories (higher index in memory_order) get higher weight
            recency_score = 0.0
            if trajectory in self.memory_order:
                recency_idx = self.memory_order.index(trajectory)
                # Normalize: most recent (last in list) = 1.0, oldest (first in list) = 0.0
                # Since memory_order is oldest to newest, we want: (idx + 1) / total
                recency_score = (recency_idx + 1) / len(self.memory_order) if len(self.memory_order) > 0 else 0.0
            else:
                # If not in memory_order, assume it's old (low recency score)
                recency_score = 0.1
            
            # Combine similarity (1 - score, since lower score = more similar) and recency
            similarity_weight = 1.0 - min(score, 1.0)  # Normalize score to 0-1
            combined_score = (1 - self.recency_weight) * similarity_weight + self.recency_weight * recency_score
            
            weighted_results.append((combined_score, trajectory))
        
        # Sort by combined score and take top-K
        weighted_results.sort(reverse=True, key=lambda x: x[0])
        top_trajectories = [traj for _, traj in weighted_results[:self.k]]
        
        return '\n'.join(top_trajectories)

    def addMemory(self, current_situation: str):
        # Create document with metadata
        memory_doc = Document(
            page_content=current_situation,
            metadata={
                "task_name": current_situation,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])
        
        # Track order (append to end = most recent)
        self.memory_order.append(current_situation)
        # Keep only recent N to avoid memory bloat
        if len(self.memory_order) > 100:
            self.memory_order = self.memory_order[-100:]


class MemoryGenerativeTopK(MemoryBase):
    """
    Memory module that uses generative importance scoring but returns top-K results.
    Combines the importance scoring of MemoryGenerative with top-K retrieval.
    """
    def __init__(self, llm, k=3):
        """
        Initialize MemoryGenerativeTopK
        
        Args:
            llm: LLM instance
            k: Number of top memories to retrieve (default: 3)
        """
        super().__init__(memory_type='generative_topk', llm=llm)
        self.k = k

    def retriveMemory(self, query_scenario: str):
        # Return empty if no memories exist
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Get more candidates for scoring
        num_candidates = min(self.k * 2, self.scenario_memory._collection.count())
        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=num_candidates)
        
        fewshot_results = []
        importance_scores = []

        # Score each memory's relevance
        for result in similarity_results:
            trajectory = result[0].metadata['task_trajectory']
            fewshot_results.append((trajectory, result[1]))  # Store with similarity score
            
            # Generate prompt to evaluate importance
            prompt = f'''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Your output format should be:
Score: '''

            # Get importance score
            try:
                response = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=10)
                score = int(re.search(r'\d+', response).group()) if re.search(r'\d+', response) else 5
            except:
                score = 5  # Default score if parsing fails
            importance_scores.append(score)

        # Combine similarity and importance scores, then select top-K
        combined_scores = []
        for idx, (trajectory, sim_score) in enumerate(fewshot_results):
            # Normalize similarity score (lower = more similar, so invert)
            normalized_sim = 1.0 - min(sim_score, 1.0)
            # Combine: 70% importance, 30% similarity
            combined = 0.7 * (importance_scores[idx] / 10.0) + 0.3 * normalized_sim
            combined_scores.append((combined, trajectory))
        
        # Sort by combined score and take top-K
        combined_scores.sort(reverse=True, key=lambda x: x[0])
        top_trajectories = [traj for _, traj in combined_scores[:self.k]]
        
        return '\n'.join(top_trajectories)
    
    def addMemory(self, current_situation: str):
        # Create document with metadata
        memory_doc = Document(
            page_content=current_situation,
            metadata={
                "task_name": current_situation,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])


class MemoryUserProfile(MemoryBase):
    """
    Memory module that builds and retrieves a user preference profile.
    Extracts patterns from all user reviews to understand rating tendencies.
    """
    def __init__(self, llm):
        """
        Initialize MemoryUserProfile
        
        Args:
            llm: LLM instance
        """
        super().__init__(memory_type='user_profile', llm=llm)
        self.user_profiles = {}  # Cache user profiles

    def retriveMemory(self, query_scenario: str):
        # Extract user_id from query if possible, or use query as-is
        # For now, we'll use the query to find similar reviews and build profile
        if self.scenario_memory._collection.count() == 0:
            return ''
        
        # Get top similar reviews to understand context
        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=5)
        
        # Extract all trajectories
        trajectories = [result[0].metadata['task_trajectory'] for result in similarity_results]
        
        # Build a simple profile from the trajectories
        if trajectories:
            # Use LLM to extract patterns
            profile_prompt = f'''Based on these review examples, identify the user's rating patterns and preferences:

Reviews:
{chr(10).join(trajectories[:3])}

Summarize the user's typical:
1. Rating range (e.g., "tends to rate 4-5 stars" or "critical, rates 2-3 stars")
2. Review style (e.g., "detailed and analytical" or "brief and emotional")
3. Focus areas (e.g., "focuses on service and quality" or "cares about price and value")

Keep it concise (2-3 sentences):'''
            
            try:
                profile = self.llm(messages=[{"role": "user", "content": profile_prompt}], temperature=0.1, max_tokens=150)
                return f"User Profile: {profile}\n\nSimilar Reviews:\n" + '\n'.join(trajectories[:2])
            except:
                return '\n'.join(trajectories[:2])
        
        return ''

    def addMemory(self, current_situation: str):
        # Create document with metadata
        memory_doc = Document(
            page_content=current_situation,
            metadata={
                "task_name": current_situation,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

