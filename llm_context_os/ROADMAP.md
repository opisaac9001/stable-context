LLM Context OS — Project Blueprint 📜
0 · Mission
Build a modular, local LLM engine that

keeps a static system prompt in context,

maintains a sliding conversation tail,

auto‑retrieves relevant “flashback” snippets (micro‑RAG),

loads GGUF, AWQ, EXL2 or remote API models interchangeably,

speaks Flash‑Attention 2 / xFormers,

supports tool‑calling (local + MCP servers),

auto‑unloads idle models,

exposes a FastAPI for scripts and a Tauri desktop GUI,

and scales to LoRA merge, speculative decoding, multimodal, etc.

1 · High‑Level Roadmap
Phase Scope Key Output Est.
0 Bootstrap Repo, CI, docs empty project skeleton ½ day
1 Context Core persistent prompt + sliding window context/ package + tests 3‑5 d
2 Model Runners GGUF, AWQ, EXL2, API runners/ + smoke tests 7‑10 d
3 Model Manager + API idle auto‑unload, gen knobs, streaming FastAPI app 4‑6 d
4 RAG + Tool‑Call chat‑history retriever, MCP client retriever.py, tool_dispatcher.py 5‑7 d
5 GUI MVP Tauri desktop (React/Svelte) chat, settings, model mgr 10‑14 d
6 Polish & Power spec‑decode, LoRA merge, multimodal docker images, auto‑tuner ∞

Total MVP: about 8 sprint‑weeks; polish continues thereafter.

2 · Filesystem Scaffold
text
Copy
Edit
llm_context_os/
├─ context/ # phase 1
│ ├─ context_manager.py
│ ├─ token_estimator.py
│ └─ __init__.py
├─ runners/ # phase 2
│ ├─ base.py
│ ├─ llama_cpp_runner.py
│ ├─ awq_runner.py
│ ├─ exl2_runner.py
│ ├─ api_runner.py
│ └─ manager.py # idle auto‑unload
├─ retriever/ # phase 4
│ └─ chat_history.py
├─ tools/ # phase 4
│ ├─ __init__.py
│ └─ builtin_weather.py
├─ api/ # phase 3
│ ├─ main.py
│ └─ schemas.py
├─ gui/ # phase 5
│ └─ … (Tauri frontend)
├─ config/
│ └─ config.yaml # idle_sec, gen default, RAG budget
├─ tests/ # every phase
├─ models/ # .gitignored
├─ requirements.txt
└─ ROADMAP.md # this file
3 · Phase Details & Skeleton Code
3.1 Context Core (context/)
context_manager.py (essential lines):

python
Copy
Edit
class ContextManager:
    def __init__(self, system_prompt, tokenizer, max_tokens=4096):
        self.system_prompt, self.tok, self.max = system_prompt, tokenizer, max_tokens
        self._messages, self._start = [], 0 # full log + window ptr

    def add(self, role, content):
        self._messages.append({"r": role, "c": content})
        self._auto_scroll()

    # window navigation
    def shift(self, steps): self._start = max(0, self._start + steps); self._fit()
    def jump_to(self, idx): self._start = max(0, min(idx, len(self._messages)-1)); self._fit()

    # prompt
    def build_prompt(self, extra=None):
        msgs = (extra or []) + self._messages[self._start:]
        return self.system_prompt + self._fmt(msgs)

    # internal trim
    def _fit(self):
        while self.tok.count_tokens(self.build_prompt()) > self.max and self._start > 0:
            self._start -= 1
    def _auto_scroll(self):
        self._start = max(0, len(self._messages)-1); self._fit()
Token estimator adapters for llama.cpp, HF, EXL2 (one per quant).

Unit tests cover: never drops system prompt, shift/jump logic, token math.

3.2 Model Runners (runners/)
base.py interface:

python
Copy
Edit
class BaseRunner(ABC):
    def generate(self, prompt: str, **cfg) -> str: ...
    def stream(self, prompt: str, **cfg) -> t.Generator[str, None, None]: ...
    def preload_kv(self, prompt: str): ... # optional
llama_cpp_runner.py loads GGUF with optional GPU layers & KV cache.

awq_runner.py uses AutoModelForCausalLM(attn_implementation="flash_attention_2").

exl2_runner.py wraps exllamav2.ExllamaModel.

api_runner.py passes chat to remote OpenAI‑style endpoint.

manager.py:

python
Copy
Edit
class ModelManager:
    async def load(self, typ, path, idle_sec=900, **kwargs): ...
    def get(self) -> BaseRunner: ... # refresh keep‑alive or raise if unloaded
3.3 FastAPI (api/main.py)
python
Copy
Edit
app = FastAPI()

@app.post("/load_model")
async def load(req: LoadModel):
    await model_mgr.load(req.type, req.path, ctx=req.ctx, gpu_layers=req.gpu)
    return {"status":"ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    ctx.add("user", req.message)
    if req.use_rag: snippets = retriever.retrieve(req.message)
    else: snippets = []
    prompt = ctx.build_prompt(snippets)
    runner = model_mgr.get()
    # stream or single reply
Streaming via sse-starlette (simple SSE) or WebSocket.

Generation knobs (temp, top_p, max_tokens, use_kv_cache) carried in ChatRequest.gen_params.

Idle auto‑unload handled by ModelManager.

3.4 Retriever + Tool Dispatcher (Phase 4)
retriever/chat_history.py
– SentenceTransformer + Chroma; 512‑token recall budget.

tool_dispatcher.py
– reads function_call JSON, executes:

local Python tools in tools/

MCP tools via langchain‑mcp‑adapters.

Adds tool result into sliding history, then resumes generation.

3.5 GUI (Phase 5)
Stack

Tauri (Rust shell) → 30 MB desktop app.

React + Vite + Tailwind (or SvelteKit).

Pages

Page Features
Chat SSE streaming, markdown render, latency & token meter
Model Manager drag‑drop GGUF/AWQ, download from TheBloke
Settings sliders: temp, top‑p, KV‑cache, idle timeout, RAG budget
Tools list MCP & local tools, toggle on/off

Persist UI state in localStorage (tauri://store for desktop).

4 · Generation & Retrieval Budget
Config defaults (config/config.yaml):

yaml
Copy
Edit
max_tokens: 4096
recent_tail_tokens: 1536
retrieval_budget_tokens: 512
idle_unload_sec: 900
defaults:
  temperature: 0.7
  top_p: 1.0
  max_tokens: 256
  use_kv_cache: true
rag:
  embed_model: all-MiniLM-L6-v2
  db_path: data/chroma
mcp_servers:
  - https://mcp.mycorp
5 · CI / Test Matrix
Layer Tests
ContextManager shift/jump/trim unit tests
Tokenizers encode/decode idempotence
Runners TinyLlama smoke gen (CPU)
API httpx ASGI client, SSE stream
Manager idle unload after 1 sec mock
Retriever recall returns ≤ budget tokens
Tools sandbox prevents file system escape

GitHub Actions jobs:
lint • unit • tiny‑model integration (CPU) • GPU (self‑hosted).

6 · Stretch & Power‑Ups (Phase 6+)
Feature Impl. hint
Speculative decoding SpecRunner wrapping draft (TinyLlama) + target (full model)
Prefix‑KV persistence save kv tensors with pickle + model‑hash key
Continuous batching / vLLM runner treat vLLM’s http server as another subclass
LoRA hot‑merge peft.inject_adapter or auto_merge before first forward
Multimodal add llava_cpp_runner, extend GUI with image upload
Vector‑PDF RAG new route /upload_doc → embed & store
Auto‑tuner first‑run benchmark chooses FA 2 vs xFormers, gpu_layers
