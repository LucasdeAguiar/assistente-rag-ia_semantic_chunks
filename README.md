# 🧠 Assistente RAG com FastAPI, ChromaDB e Extração de Assinatura

Este projeto é uma evolução robusta da base RAG com FastAPI, incorporando:

- Segmentação **semântica de texto com LLM**
- Extração automática do nome de quem assinou o documento
- Classificação temática de trechos
- Armazenamento vetorial otimizado com metadados
- Re-ranking manual por similaridade de cosseno

---

## 🚀 Novidades desta versão

- 🧠 **Segmentação semântica com LLM (gpt-3.5-turbo)** para dividir o texto em *chunks* por tópico
- ✍️ **Extração automática de assinatura** (regex + fallback com LLM)
- 🔖 **Classificação temática** dos chunks (ex: assinatura, identificação, benefícios, dependentes, etc.)
- 🔁 **Fallback com LLM** para documentos mal formatados
- 🧪 **Re-ranking manual** por similaridade de cosseno (cosine similarity)
- 🔍 **Filtro por descrição temática** nas buscas RAG (ex: buscar só chunks de assinatura)
- 🧹 **Endpoint para limpar a base** com facilidade

---

## 🛠 Tecnologias utilizadas

| Componente             | Ferramenta                                     |
|------------------------|------------------------------------------------|
| 🔡 Embeddings          | OpenAI `text-embedding-3-small`                |
| 🧠 LLM                 | GPT-3.5-turbo (classificação + fallback)       |
| 📦 Banco vetorial      | ChromaDB (modo persistente)                    |
| 📄 Leitura de PDF      | PyMuPDF (`fitz`)                               |
| ⚙️ Backend             | FastAPI                                        |
| 🤖 Geração de resposta | Ollama (modelo `llama3` rodando localmente)    |

---

## ✅ Funcionalidades

- [x] Upload de texto manual ou via PDF
- [x] Segmentação semântica dos documentos com LLM
- [x] Geração de embedding para cada chunk
- [x] Classificação temática de chunks com LLM
- [x] Extração do nome da assinatura (regex + LLM)
- [x] Armazenamento vetorial com metadados enriquecidos
- [x] Busca com RAG + re-ranking por similaridade
- [x] Filtro por tema (ex: `assinatura`)
- [x] Listagem completa dos documentos e metadados
- [x] Endpoint para **limpeza total da base**

---

## ▶️ Como rodar localmente

```bash
python -m venv venv
venv\Scripts\activate  # ou source venv/bin/activate no Linux/macOS
pip install -r requirements.txt
uvicorn app.main:app --reload
