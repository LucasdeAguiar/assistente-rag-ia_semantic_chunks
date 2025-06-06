Assistente RAG com FastAPI, ChromaDB e Extração de Assinatura
Este projeto é uma evolução robusta da base RAG com FastAPI, incorporando técnicas de segmentação semântica com LLM, extração de nome de assinatura, e classificação temática automática de trechos do documento.

🚀 Novidades desta versão

🧠 Segmentação semântica com LLM (gpt-3.5-turbo) para dividir o texto em chunks coesos por tópico

✍️ Extração automática do nome da pessoa que assinou o documento, via regex e fallback com LLM

🔖 Classificação temática de cada chunk, como: assinatura, identificação, valores, benefícios, dependentes, etc.

🧠 Fallback com LLM para inferência de nome em documentos com formatação não padronizada

🧪 Re-ranking manual com similaridade de cosseno

🔎 Filtragem por descrição temática na consulta RAG (ex: buscar apenas em chunks classificados como assinatura)

🧹 Endpoint para limpeza total da base

🧠 Tecnologias utilizadas

🔡 Embeddings: OpenAI text-embedding-3-small

🧠 LLM (classificação e fallback): gpt-3.5-turbo

🗂 Banco vetorial: ChromaDB (modo persistente)

📄 Leitura de PDF: PyMuPDF (fitz)

⚙️ Backend: FastAPI

🤖 Geração de resposta: Ollama (LLaMA3) rodando localmente

✅ Funcionalidades

 Upload de texto manual ou por PDF

 Segmentação semântica de chunks com LLM

 Geração de embedding para cada chunk

 Classificação temática de trechos com LLM

 Extração automática do nome da assinatura (regex + LLM)

 Armazenamento vetorial com metadados enriquecidos

 Consulta com RAG + re-ranking manual

 Filtro por descrição temática nas buscas

 Listagem dos documentos e metadados

 Limpeza da base


▶️ Como rodar localmente

python -m venv venv
venv\Scripts\activate  # ou source venv/bin/activate no Linux/macOS
pip install -r requirements.txt
uvicorn app.main:app --reload

📌 Requisitos

Python 3.10+

Ollama rodando localmente com o modelo llama3

ChromaDB configurado em modo persistente (./chroma)


💡 Exemplo de uso

Envie um PDF com uma declaração assinada

A API extrai o texto, segmenta semanticamente e classifica os chunks

O nome do signatário é extraído e vinculado ao chunk com descrição assinatura, identificação

A API pode responder perguntas como:

"Quem assinou este documento?"

"Quais são os benefícios descritos?"

"Existe seção sobre vale transporte?"

