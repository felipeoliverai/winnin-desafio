# Scientific Q&A Agent (RAG)

Agente de Q&A sobre 3 artigos clássicos de ML — *Attention Is All You Need*, *BERT* e *RAG* — exposto via API REST FastAPI. Usa **RAG sobre ChromaDB** + **Gemini 2.0 Flash** com **function calling** nativo.

---

## 1. Visão geral da arquitetura

```
            ┌─────────────────┐
  Usuário → │  POST /ask      │  (FastAPI)
            └────────┬────────┘
                     ▼
            ┌─────────────────────────────┐
            │        QAAgent              │
            │  Gemini 2.0 Flash           │
            │  + function calling         │
            ├─────────────────────────────┤
            │ Tools:                      │
            │  • search_documents         │──┐
            │  • extract_section          │──┤
            └─────────────────────────────┘  │
                                             ▼
                                  ┌────────────────────┐
                                  │    ChromaDB         │
                                  │ (vector store local)│
                                  └─────────┬───────────┘
                                            ▲
                              ingest.py: PDFs → chunking → embeddings
```

**Fluxo de uma pergunta:**
1. `POST /ask` recebe a pergunta.
2. `QAAgent` envia para o Gemini junto com o schema das tools.
3. Gemini decide chamar `search_documents` (ou `extract_section`) com argumentos.
4. O agente executa a tool localmente (consulta ChromaDB) e devolve o resultado ao Gemini.
5. Gemini consolida a resposta final e cita o(s) paper(s).

---

## 2. Distinção entre tools e agente

| Conceito | Responsabilidade | Implementação |
|---|---|---|
| **Tool** | Operação atômica, sem memória, sem decisões. Recebe parâmetros tipados, retorna `ToolResult`. | [`app/tools.py`](app/tools.py) — classes `SearchDocuments` e `ExtractSection` herdam de `Tool` (Pydantic). |
| **Agente** | Recebe pergunta, decide quais tools chamar (via function calling do Gemini), itera até produzir a resposta final. | [`app/agent.py`](app/agent.py) — classe `QAAgent` com loop de no máximo 5 iterações de tool call. |

As tools **não conhecem o LLM** — são puramente funções tipadas sobre o vector store. Trocar o Gemini por outro provider só exige reescrever `QAAgent`.

---

## 3. Setup

### Opção A — Docker (recomendado)

```bash
cd winnin-desafio
cp .env.example .env
# preencha GEMINI_API_KEY no .env

docker compose up --build
```

A primeira subida automaticamente baixa os 3 PDFs e popula o ChromaDB (entrypoint detecta vector store vazio). Os dados ficam em `./data/` via volume — restarts subsequentes pulam a ingestão.

### Opção B — Local

```bash
cd winnin-desafio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# preencha GEMINI_API_KEY no .env

python ingest.py             # baixa os 3 PDFs e indexa no ChromaDB
uvicorn app.main:app         # sobe a API em http://localhost:8000
```

Documentação interativa em `http://localhost:8000/docs`.

### Exemplo

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "O que é RAG e quais problemas ele resolve segundo os autores?"}'
```

### Testes

```bash
pytest -v
```

---

## 4. Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| **LLM** | Gemini Flash (2.5 por padrão; 2.0 também suportado) | Requisito do desafio. Suporte nativo a function calling. O default é `gemini-2.5-flash` porque é o que tem free tier ativo na maior parte das chaves novas do AI Studio — basta trocar `GEMINI_MODEL` no `.env` para `gemini-2.0-flash` se a chave tiver quota lá. |
| **SDK** | `google-generativeai` direto | Sem LangChain/LlamaIndex — código explícito, fácil de testar e mockar, sem abstrações que ocultam o function calling. |
| **Vector store** | ChromaDB persistente | Setup zero, persiste em disco automaticamente, API simples e suporta filtros por metadata (necessário para `extract_section`). |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Roda local, zero custo de API, qualidade adequada para textos científicos curtos em inglês. |
| **Parser de PDF** | `pypdf` | Lib pequena, sem dependências nativas, suficiente para os 3 papers. |
| **Chunking** | Recursive char split (~800 chars, overlap 150) + detecção de seção por regex | Padrão estável para artigos científicos; preserva contexto entre chunks e permite filtragem por seção. |
| **Settings** | `pydantic-settings` + `.env` | Requisito do desafio. |
| **Validação** | Pydantic v2 em todos os modelos (input, output, chunks, tool results) | Requisito do desafio. |

---

## 5. Limitações conhecidas

- **Sem reranking**: a recuperação usa apenas similaridade de cosseno do MiniLM. Para perguntas ambíguas, um cross-encoder melhoraria o ranking.
- **Detecção de seção heurística**: regex sobre headings funciona bem para os 3 papers, mas pode classificar errado se o PDF tiver layout incomum. A seção é "best effort" — `search_documents` continua funcionando como fallback.
- **Embeddings em inglês**: o modelo escolhido é mais forte em EN. Perguntas em PT continuam funcionando (a busca é semântica), mas o recall pode ser ligeiramente menor que com um modelo multilíngue.
- **Sem cache de respostas**: cada `/ask` chama o Gemini do zero.
- **Sem autenticação na API**: pensada para rodar localmente.
- **Parser de PDF perde tabelas e figuras**: como esperado de `pypdf`. Conteúdo textual dos 3 papers é suficiente para as 3 perguntas-alvo.
- **Sem streaming na resposta**: `/ask` é síncrono.
- **Rate limit do free tier do Gemini**: 5 req/min. Há retry com backoff respeitando o `retry_delay` sugerido pela API, mas perguntas em rajada podem ainda assim falhar.

---

## 6. Estrutura

```
winnin-desafio/
├── app/
│   ├── main.py        # FastAPI + lifespan
│   ├── config.py      # pydantic-settings
│   ├── models.py      # Pydantic v2 (Chunk, ToolResult, AskRequest/Response)
│   ├── chunking.py    # split + section detection
│   ├── rag.py         # ChromaDB wrapper
│   ├── tools.py       # SearchDocuments, ExtractSection
│   └── agent.py       # QAAgent (Gemini function-calling loop)
├── ingest.py          # download + index
├── tests/             # pytest (chunking, tools, agent)
├── requirements.txt
├── .env.example
└── README.md
```
