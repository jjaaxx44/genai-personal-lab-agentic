from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .budget import BudgetDefaults

# app/core/config.py -> app/ -> the repo root, which holds samples/ and data/.
# In the container that resolves to /app, where samples/ is copied and ./data is mounted.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM providers, tried in this order: Gemini -> Groq -> OpenAI -> local Ollama.
    # A provider joins the chain only when both its key and its model are set.
    gemini_api_key: str = ""
    gemini_llm_model: str = ""
    gemini_llm_fast_model: str = ""

    groq_api_key: str = ""
    groq_llm_model: str = ""
    groq_llm_fast_model: str = ""

    openai_api_key: str = ""
    openai_llm_model: str = ""
    openai_llm_fast_model: str = ""

    # Ollama runs on the host, not in Docker, and serves one model -- no fast variant.
    local_ollama_base_url: str = "http://host.docker.internal:11434"
    local_ollama_llm_model: str = ""

    # Local models
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # MongoDB Atlas Local
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "genai_agentic_lab"

    # Langfuse (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Budgets. Every run is bounded by all three (ground rule 2). These are the
    # defaults the sidebar starts from; a run can lower or raise them per run.
    agent_max_steps: int = 12
    agent_max_tokens: int = 60_000
    agent_deadline_s: float = 180

    # Tools
    tool_timeout_s: float = 30
    web_search_results: int = 4
    sql_row_limit: int = 50
    corpus_top_k: int = 5
    corpus_chunk_size: int = 800
    corpus_chunk_overlap: int = 100
    vfs_root: str = "data/vfs"
    vfs_max_file_kb: int = 256
    vfs_max_files: int = 50

    # Demo-specific caps (one per demo as it lands)
    reflexion_max_attempts: int = 3
    plan_execute_max_plan_steps: int = 6
    # A replan costs one model call per executed step (execute + replan), plus the
    # plan and respond calls, so this demo's own step budget starts higher than the
    # shared agent_max_steps default -- see Step 3's implementation brief.
    plan_execute_max_steps: int = 20
    supervisor_max_hops: int = 8
    swarm_max_handoffs: int = 6
    subagent_max_steps: int = 6
    memory_recall_k: int = 5
    autonomous_max_objectives: int = 8
    research_max_subquestions: int = 4
    eval_task_set_size: int = 8

    def budget_defaults(self) -> BudgetDefaults:
        return BudgetDefaults(
            max_steps=self.agent_max_steps,
            max_tokens=self.agent_max_tokens,
            deadline_s=self.agent_deadline_s,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
