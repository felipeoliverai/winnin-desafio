# Plano de Execução — Desafio Data Science RAG

> Plano de implementação para o desafio descrito em [DATASCI_RAG.md](../DATASCI_RAG.md): construir um agente de Q&A sobre 3 artigos científicos (Attention Is All You Need, BERT, RAG), exposto via API FastAPI, com tools, function calling e RAG sobre vector store local.

---

## 1. Decisões técnicas (a justificar no README)

| Tópico | Escolha | Motivo |
|---|---|---|
| LLM | **Gemini 2.0 Flash** (obrigatório) via `google-generativeai` | Suporte nativo a function calling, gratuito, sem dependência extra de framework pesado |
| Framework de agente | **`google-generativeai` direto** (sem LangChain/LlamaIndex) | Mantém o código explícito, fácil de testar, sem abstrações que escondem o function calling |
| Vector Store | **ChromaDB** (persistente em disco) | Setup zero, persistência local automática, API simples |
| Embeddings | **`sentence-transformers/all-MiniLM-L6-v2`** | Roda local, sem custo de API, qualidade boa para textos científicos curtos |
| Parser de PDF | **`pypdf`** (ou `pymupdf` se qualidade ruim) | Lib pequena, sem dependências nativas pesadas |
| Chunking | **Recursive character splitting** (~800 chars, overlap 150) | Padrão estável para artigos científicos; preserva contexto entre chunks |
| Settings | **`pydantic-settings`** com `.env` | Requisito explícito do desafio |
| Testes | **`pytest`** + `pytest-mock` | Requisito explícito do desafio |

---

## 2. Estrutura do projeto

```
winnin-desafio/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, monta router
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # POST /ask
│   │   └── schemas.py       # AskRequest, AskResponse (Pydantic v2)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── qa_agent.py      # Agente Gemini com function calling
│   │   └── prompts.py       # System prompt do agente
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py          # Classe Tool abstrata + ToolResult
│   │   ├── search_documents.py
│   │   └── extract_section.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_store.py  # Wrapper Chroma
│   │   ├── embeddings.py    # Wrapper sentence-transformers
│   │   └── chunking.py      # Splitter
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # pydantic-settings
│   │   └── logging.py       # setup logging
│   └── models/
│       ├── __init__.py
│       └── domain.py        # Chunk, Paper, Section (Pydantic v2)
├── ingest.py                # Script standalone: baixa PDFs + popula Chroma
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_search_documents.py
│   ├── test_extract_section.py
│   ├── test_chunking.py
│   └── test_agent.py        # mocka Gemini
├── data/
│   ├── pdfs/                # PDFs baixados (gitignored)
│   └── chroma/              # base vetorial (gitignored)
├── .env.example
├── requirements.txt
├── pyproject.toml           # ruff + pytest config
└── README.md
```

---

## 3. Modelagem de dados (Pydantic v2)

```python
# app/models/domain.py
class Paper(BaseModel):
    arxiv_id: str
    title: str
    pdf_path: Path

class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    section: str | None
    text: str
    page: int | None

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Any
    error: str | None = None

# app/api/schemas.py
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]   # paper IDs/títulos consultados
```

---

## 4. Tools — contrato

Classe base `Tool` com:
- `name: str` (class attr)
- `description: str` (class attr — vai pro Gemini)
- `parameters_schema: dict` (formato function declaration do Gemini)
- `run(**kwargs) -> ToolResult` (método único)

### `search_documents`
- **Input**: `query: str`, `top_k: int = 4`
- **Output**: lista de chunks ranqueados (texto, paper, score)
- **Uso**: busca semântica geral na base Chroma

### `extract_section`
- **Input**: `paper_id: Literal["1706.03762", "1810.04805", "2005.11401"]`, `section: Literal["abstract", "introduction", "conclusion", ...]`
- **Output**: texto da seção do paper
- **Implementação**: durante a ingestão, marcar chunks com label de seção (regex sobre headings: `Abstract`, `1 Introduction`, `Conclusion`, etc.). A tool filtra por `paper_id + section` no Chroma via metadata filter.

---

## 5. Pipeline de ingestão (`ingest.py`)

1. Baixar 3 PDFs do arXiv (`https://arxiv.org/pdf/{id}.pdf`) para `data/pdfs/` — pular se já existe
2. Para cada PDF:
   1. Extrair texto por página com `pypdf`
   2. Detectar seções por regex de heading
   3. Chunk recursivo (~800 chars, overlap 150)
   4. Anexar metadata: `paper_id`, `paper_title`, `section`, `page`
3. Embeddings em batch com `sentence-transformers`
4. Persistir em ChromaDB (`data/chroma/`)
5. Logar contagens (papers, chunks, dimensão dos embeddings)

Idempotente: se a coleção já existe e tem N chunks esperados, pular reingestão (com flag `--force` para reindexar).

---

## 6. Agente Q&A

Loop de function calling com Gemini 2.0 Flash:

1. Recebe `question`
2. Chama `model.generate_content` com tools declaradas (`search_documents`, `extract_section`)
3. Se Gemini retorna `function_call` → executa a tool localmente → devolve `function_response` no histórico → repete
4. Quando Gemini retorna texto final → retorna ao usuário
5. Limite duro de iterações (ex: 5) pra evitar loop infinito

System prompt (resumido): "Você responde perguntas APENAS com base nos 3 artigos indexados. Use `search_documents` para busca semântica e `extract_section` quando a pergunta pedir uma seção específica. Cite o paper. Se não encontrar a resposta, diga que não encontrou."

---

## 7. API FastAPI

- `POST /ask` → `AskRequest` → `AskResponse`
- `GET /health` → status simples (extra, não obrigatório)
- Swagger automático em `/docs`
- Dependency injection do agente (singleton inicializado no `lifespan`)
- Tratamento de erros: `HTTPException` para input inválido, 500 com mensagem genérica para erros internos (logados)

---

## 8. Configuração e qualidade

- **`config.py`**: `GEMINI_API_KEY`, `CHROMA_PATH`, `EMBEDDING_MODEL`, `TOP_K`, `LOG_LEVEL` via `pydantic-settings`
- **`.env.example`** comentado com cada variável
- **Logging** estruturado (`logging.getLogger(__name__)`) — sem `print`
- **Type hints** em tudo
- **Ruff** configurado em `pyproject.toml`
- Sem `except Exception: pass` — exceções tratadas explicitamente nas tools (retornam `ToolResult(success=False, error=...)`)

---

## 9. Testes (mínimo)

- `test_chunking.py` — splitting, overlap, detecção de seções
- `test_search_documents.py` — mocka Chroma, valida formato de saída
- `test_extract_section.py` — mocka Chroma com filtro de metadata
- `test_agent.py` — mocka Gemini, valida loop de function calling (1 chamada de tool → resposta final)
- `conftest.py` com fixtures de chunks de exemplo

Meta: cobertura suficiente para validar contratos das tools e do agente sem chamar Gemini de verdade.

---

## 10. README — checklist obrigatório

1. Diagrama textual do fluxo Usuário → API → Agente → Tools → Vector Store
2. Distinção tools vs agente (com snippet curto)
3. Setup passo a passo: `pip install`, `cp .env.example .env`, preencher `GEMINI_API_KEY`, `python ingest.py`, `uvicorn app.main:app --reload`, exemplo `curl POST /ask`
4. Decisões técnicas (tabela da seção 1 deste plano)
5. Limitações conhecidas: sem reranking, sem cache de respostas, embeddings em inglês, parser de PDF pode falhar em tabelas/figuras, sem autenticação na API

---

## 11. Cronograma sugerido (7 dias)

| Dia | Entrega |
|---|---|
| 1 | Estrutura do projeto, config, modelos Pydantic, `.env.example`, `requirements.txt` |
| 2 | `ingest.py` funcional: download + parsing + chunking + embeddings + Chroma |
| 3 | Tools (`search_documents`, `extract_section`) + testes unitários |
| 4 | Agente Q&A com loop de function calling do Gemini |
| 5 | API FastAPI + integração end-to-end + teste manual das 3 perguntas |
| 6 | Testes de agente (mockados), polish de logging/erros, ruff clean |
| 7 | README completo, revisão final, push do repo público |

---

## 12. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Parser de PDF perdendo estrutura | Fallback `pymupdf` se `pypdf` falhar; testar ingestão antes de avançar |
| Function calling do Gemini instável | Limitar iterações; fallback para chamada direta com contexto recuperado |
| Embeddings ruins p/ termos técnicos | Top-k generoso (4-6); validar manualmente as 3 perguntas-alvo no fim do dia 2 |
| Quota gratuita do Gemini | Usar cache local da resposta durante dev; rodar suite de testes mockada |
