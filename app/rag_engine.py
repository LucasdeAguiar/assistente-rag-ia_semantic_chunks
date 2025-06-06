# rag_engine.py

import os
import re
import fitz
import numpy as np
import openai
import requests
import chromadb
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
print("Chave carregada:", openai.api_key[:10], "...")

chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="base_conhecimento")

def gerar_embedding(texto: str) -> list[float]:
    response = openai.embeddings.create(
        input=texto,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def extrair_texto_pdf(caminho_pdf: str) -> str:
    doc = fitz.open(caminho_pdf)
    texto = "".join(pagina.get_text() for pagina in doc)
    doc.close()
    return texto

def segmentar_semanticamente(texto: str) -> list[str]:
    prompt = f"""
Divida o texto abaixo em seções temáticas bem definidas. Cada seção deve conter apenas um tópico coeso.
Separe com ###.

Texto:
{texto}
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content
    return [chunk.strip() for chunk in raw.split("###") if chunk.strip()]

def descrever_chunk(chunk: str) -> str:
    padroes = [
        r"\beu[, ]", r"declaro.*?recebido", r"portador do RG", r"assinatura",
        r"nome do profissional", r"assinatura do.*(profissional|colaborador|responsável)"
    ]
    for padrao in padroes:
        if re.search(padrao, chunk.lower()):
            return "assinatura, identificação"

    prompt = f"""
Classifique o trecho abaixo com um dos temas:
- assinatura, identificação
- plano odontológico
- plano de saúde
- valores, benefícios
- vale transporte
- dependentes, inclusão
- outros

Trecho:
"""
    prompt += chunk

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()

def extrair_nome_assinatura(texto: str) -> str:
    match = re.search(r"eu[\s,:-]+(.*?)(?:,| portador|$)", texto, re.IGNORECASE)
    if match:
        nome = match.group(1).strip()
        if len(nome.split()) >= 2 and all(p.isalpha() for p in nome.replace(" ", "")):
            return nome

    match = re.search(r"([\w\s]{5,})\s+portador do RG", texto, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return fallback_nome_por_llm(texto)

def fallback_nome_por_llm(texto: str) -> str:
    prompt = f"""
Extraia apenas o nome completo da pessoa, se houver uma frase:
"Eu {{NOME}}, portador do RG"

Texto:
{texto}

Nome:
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def adicionar_texto(texto: str, origem: str = "manual", nome_arquivo: str = "", caminho_pdf: str = None):
    nome_assinatura = extrair_nome_assinatura(texto)
    id_base = len(collection.get()["ids"])
    chunks = segmentar_semanticamente(texto)

    for i, chunk in enumerate(chunks):
        embedding = gerar_embedding(chunk)
        descricao = descrever_chunk(chunk)
        metadados = {
            "origem": origem,
            "arquivo": nome_arquivo,
            "descricao": descricao
        }
        if descricao == "assinatura, identificação" and nome_assinatura:
            metadados["nome_assinatura"] = nome_assinatura

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"id-{id_base + i + 1}-{i}"],
            metadatas=[metadados]
        )
        print(f"[OK] Chunk {i} salvo com descrição: {descricao}")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def buscar_resposta(pergunta: str, top_c: int = 3, descricao_filtrada: str = None) -> str:
    embedding_pergunta = gerar_embedding(pergunta)
    resultados = collection.get(include=["documents", "embeddings", "metadatas"])
    documentos, embeddings, metadatas = resultados["documents"], resultados["embeddings"], resultados["metadatas"]

    if descricao_filtrada:
        documentos, embeddings, metadatas = zip(*[
            (doc, np.array(emb), meta)
            for doc, emb, meta in zip(documentos, embeddings, metadatas)
            if descricao_filtrada.lower() in meta.get("descricao", "").lower()
        ]) if documentos else ([], [], [])

    for meta in metadatas:
        if meta.get("descricao") == "assinatura, identificação" and meta.get("nome_assinatura"):
            return f"O documento foi assinado por {meta['nome_assinatura']}."

    scores = [cosine_similarity(embedding_pergunta, emb) for emb in embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_c]
    contexto = "\n".join([documentos[i] for i in top_indices])

    prompt = f"Responda com base no seguinte contexto:\n{contexto}\n\nPergunta: {pergunta}"
    return gerar_resposta_ollama(prompt)

def gerar_resposta_ollama(prompt: str, modelo: str = "llama3") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": modelo, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"]

def listar_dados_salvos():
    return collection.get(include=["documents", "metadatas"])

def limpar_base():
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
