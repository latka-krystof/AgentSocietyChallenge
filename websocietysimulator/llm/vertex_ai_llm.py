"""
Google Cloud Vertex AI LLM implementation for websocietysimulator.
Uses GCP credits - perfect if you have free GCP credits!

To use:
1. Set up GCP project with Vertex AI enabled
2. Authenticate: gcloud auth application-default login
3. Install: pip install google-cloud-aiplatform
4. Use: simulator.set_llm(VertexAILLM(project_id="your-project", location="us-central1"))
"""
from typing import Dict, List, Optional, Union
from .llm import LLMBase
import logging

logger = logging.getLogger("websocietysimulator")

try:
    from vertexai.generative_models import GenerativeModel
    from google.cloud import aiplatform
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logger.warning("Vertex AI not available. Install with: pip install google-cloud-aiplatform")


class VertexAILLM(LLMBase):
    """
    Google Cloud Vertex AI LLM implementation.
    Uses GCP credits - great if you have free GCP credits!
    
    Supports:
    - gemini-2.5-pro (latest, recommended)
    - gemini-2.5-flash (fast, cheaper)
    - gemini-2.5-flash-image (with vision)
    - gemini-2.5-flash-lite (lightweight)
    - gemini-2.0-flash-001
    - gemini-2.0-flash-lite-001
    - gemini-pro (legacy)
    - gemini-1.5-pro (legacy)
    - gemini-1.5-flash (legacy)
    - text-bison (PaLM)
    - chat-bison (PaLM)
    """
    
    def __init__(
        self, 
        project_id: str, 
        location: str = "us-central1",
        model: str = "gemini-2.5-pro",
        credentials_path: Optional[str] = None,
        use_vertex_ai_embeddings: bool = False,
        embedding_model: str = "text-embedding-004"
    ):
        """
        Initialize Vertex AI LLM
        
        Args:
            project_id: Your GCP project ID
            location: GCP region (us-central1, us-east1, etc.)
            model: Model name (gemini-2.5-pro, gemini-2.5-flash, etc.)
            credentials_path: Optional path to service account JSON (or use default credentials)
            use_vertex_ai_embeddings: If True, use Vertex AI embeddings. Default False (uses sentence-transformers).
            embedding_model: Vertex AI embedding model name (text-embedding-004, text-embedding-005, gemini-embedding-001)
        """
        if not VERTEX_AI_AVAILABLE:
            raise ImportError(
                "Vertex AI not available. Install with: "
                "pip install google-cloud-aiplatform"
            )
        
        super().__init__(model)
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Initialize model
        self.model_instance = GenerativeModel(model)
        
        # Initialize embeddings - defaults to sentence-transformers, optionally use Vertex AI
        self.embedding_model = VertexAIEmbeddingWrapper(
            project_id=project_id,
            location=location,
            use_vertex_ai=use_vertex_ai_embeddings,
            model_name=embedding_model
        )
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None, 
        temperature: float = 0.0, 
        max_tokens: int = 500, 
        stop_strs: Optional[List[str]] = None, 
        n: int = 1
    ) -> Union[str, List[str]]:
        """
        Call Vertex AI API to get response
        
        Args:
            messages: List of input messages
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stop_strs: Optional list of stop strings
            n: Number of responses to generate
            
        Returns:
            Union[str, List[str]]: Response text from LLM
        """
        if not VERTEX_AI_AVAILABLE:
            raise ImportError("Vertex AI not available")
        
        try:
            # Convert messages to Gemini format
            # Vertex AI Gemini uses a different format than OpenAI
            prompt = self._messages_to_prompt(messages)
            
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            if stop_strs:
                generation_config["stop_sequences"] = stop_strs
            
            if n == 1:
                # Single response
                response = self.model_instance.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text
            else:
                # Multiple responses
                responses = []
                for _ in range(n):
                    response = self.model_instance.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    responses.append(response.text)
                return responses
                
        except Exception as e:
            logger.error(f"Vertex AI API Error: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning("Vertex AI API rate limit or quota exceeded")
            raise e
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to Vertex AI prompt format
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            str: Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                # Vertex AI doesn't have system messages, prepend to first user message
                if prompt_parts:
                    prompt_parts[0] = f"System: {content}\n\n{prompt_parts[0]}"
                else:
                    prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)
    
    def get_embedding_model(self):
        """
        Get the embedding model for text embeddings
        
        Returns:
            VertexAIEmbeddings: An instance of Vertex AI embedding model
        """
        return self.embedding_model


class VertexAIPaLMLLM(LLMBase):
    """
    Alternative: Vertex AI PaLM API (text-bison, chat-bison)
    Uses GCP credits - another option if you prefer PaLM models
    """
    
    def __init__(
        self, 
        project_id: str, 
        location: str = "us-central1",
        model: str = "text-bison@001",
        use_vertex_ai_embeddings: bool = False,
        embedding_model: str = "text-embedding-004"
    ):
        """
        Initialize Vertex AI PaLM LLM
        
        Args:
            project_id: Your GCP project ID
            location: GCP region
            model: Model name (text-bison@001, chat-bison@001)
            use_vertex_ai_embeddings: If True, use Vertex AI embeddings. Default False (uses sentence-transformers).
            embedding_model: Vertex AI embedding model name (text-embedding-004, text-embedding-005, gemini-embedding-001)
        """
        if not VERTEX_AI_AVAILABLE:
            raise ImportError(
                "Vertex AI not available. Install with: "
                "pip install google-cloud-aiplatform"
            )
        
        super().__init__(model)
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Import PaLM model
        from vertexai.preview.language_models import TextGenerationModel
        
        self.model_instance = TextGenerationModel.from_pretrained(model)
        self.embedding_model = VertexAIEmbeddingWrapper(
            project_id=project_id,
            location=location,
            use_vertex_ai=use_vertex_ai_embeddings,
            model_name=embedding_model
        )
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None, 
        temperature: float = 0.0, 
        max_tokens: int = 500, 
        stop_strs: Optional[List[str]] = None, 
        n: int = 1
    ) -> Union[str, List[str]]:
        """Call Vertex AI PaLM API"""
        try:
            prompt = self._messages_to_prompt(messages)
            
            response = self.model_instance.predict(
                prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                stop_sequences=stop_strs if stop_strs else None,
            )
            
            if n == 1:
                return response.text
            else:
                # Generate multiple responses
                responses = []
                for _ in range(n):
                    resp = self.model_instance.predict(
                        prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        stop_sequences=stop_strs if stop_strs else None,
                    )
                    responses.append(resp.text)
                return responses
                
        except Exception as e:
            logger.error(f"Vertex AI PaLM Error: {e}")
            raise e
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt"""
        prompt_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        return "\n".join(prompt_parts)
    
    def get_embedding_model(self):
        return self.embedding_model


class VertexAIEmbeddingWrapper:
    """
    Wrapper for embeddings that works with langchain-chroma.
    Defaults to sentence-transformers (fast, free, local).
    Optionally uses Vertex AI embeddings if use_vertex_ai=True.
    
    Available Vertex AI embedding models:
    - text-embedding-004 (recommended)
    - text-embedding-005 (latest)
    - gemini-embedding-001 (Gemini-based)
    - text-multilingual-embedding-002 (multilingual)
    """
    def __init__(
        self,
        project_id: str = None,
        location: str = None,
        use_vertex_ai: bool = False,
        model_name: str = "text-embedding-004"
    ):
        """
        Initialize embedding wrapper.
        
        Args:
            project_id: GCP project ID (required if use_vertex_ai=True)
            location: GCP location (required if use_vertex_ai=True)
            use_vertex_ai: If True, use Vertex AI embeddings. Default False (sentence-transformers).
            model_name: Vertex AI embedding model name (text-embedding-004, text-embedding-005, gemini-embedding-001)
        """
        self.use_vertex_ai = use_vertex_ai
        self.vertex_ai_model = None
        
        # Default to sentence-transformers
        from sentence_transformers import SentenceTransformer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model = self.sentence_transformer
        
        if use_vertex_ai:
            if not project_id or not location:
                raise ValueError("project_id and location are required when use_vertex_ai=True")
            
            try:
                from vertexai.language_models import TextEmbeddingModel
                
                # Try the specified model
                models_to_try = [
                    model_name,
                    "text-embedding-004",
                    "text-embedding-005",
                    "gemini-embedding-001"
                ]
                
                model_loaded = False
                for model_to_try in models_to_try:
                    try:
                        test_model = TextEmbeddingModel.from_pretrained(model_to_try)
                        # Test if it works
                        test_embeddings = test_model.get_embeddings(["test"])
                        if test_embeddings:
                            self.vertex_ai_model = test_model
                            self.embedding_model = self.vertex_ai_model
                            logger.info(f"Using Vertex AI embedding model: {model_to_try}")
                            model_loaded = True
                            break
                    except Exception as e:
                        logger.debug(f"Model {model_to_try} not available: {e}")
                        continue
                
                if not model_loaded:
                    logger.warning(
                        "Vertex AI embeddings not available, falling back to sentence-transformers"
                    )
                    self.use_vertex_ai = False
                    
            except (ImportError, Exception) as e:
                logger.warning(
                    f"Vertex AI embeddings not available ({type(e).__name__}), "
                    "using sentence-transformers"
                )
                self.use_vertex_ai = False
        else:
            logger.info("Using sentence-transformers for embeddings (default)")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if self.use_vertex_ai and self.vertex_ai_model:
            try:
                embeddings = self.vertex_ai_model.get_embeddings(texts)
                return [emb.values for emb in embeddings]
            except Exception as e:
                logger.warning(f"Vertex AI embedding failed: {e}, using sentence-transformers")
                self.use_vertex_ai = False
                return self.sentence_transformer.encode(texts).tolist()
        else:
            return self.sentence_transformer.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if self.use_vertex_ai and self.vertex_ai_model:
            try:
                embeddings = self.vertex_ai_model.get_embeddings([text])
                return embeddings[0].values if embeddings else []
            except Exception as e:
                logger.warning(f"Vertex AI embedding failed: {e}, using sentence-transformers")
                self.use_vertex_ai = False
                return self.sentence_transformer.encode([text])[0].tolist()
        else:
            return self.sentence_transformer.encode([text])[0].tolist()
    
    def __call__(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Callable interface for compatibility"""
        if isinstance(texts, str):
            return self.embed_query(texts)
        else:
            return self.embed_documents(texts)
