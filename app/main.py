from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from typing import Optional
import shutil
import os
import uuid

from app.rag_engine import (
    adicionar_texto,
    extrair_texto_pdf,
    buscar_resposta,
    limpar_base,
    listar_dados_salvos
)
from app.modelos import TextoEntrada, PerguntaEntrada

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG rodando"}

@app.post("/upload")
def upload_texto(payload: TextoEntrada):
    try:
        adicionar_texto(payload.texto, origem="manual", nome_arquivo="inserido_manual")
        return {"message": "Texto armazenado com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    try:
        nome_arquivo = file.filename
        caminho_temp = f"temp_{uuid.uuid4()}.pdf"

        with open(caminho_temp, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        texto = extrair_texto_pdf(caminho_temp)
        adicionar_texto(texto, origem="pdf", nome_arquivo=nome_arquivo)

        os.remove(caminho_temp)

        return {"message": f"PDF '{nome_arquivo}' processado com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pergunta")
def responder_pergunta(
    payload: PerguntaEntrada,
    top_c: int = Query(3, ge=1, le=10),
    descricao: Optional[str] = Query(None, description="Filtrar chunks por descrição")
):
    try:
        resposta = buscar_resposta(payload.pergunta, top_c=top_c, descricao_filtrada=descricao)
        return {"resposta": resposta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/listar")
def listar():
    dados = listar_dados_salvos()
    documentos_filtrados = [
        {"document": dados["documents"][i], "metadata": dados["metadatas"][i]}
        for i in range(len(dados["documents"]))
    ]
    return JSONResponse(content=documentos_filtrados)

@app.post("/limpar")
def limpar_collection():
    try:
        limpar_base()
        return {"message": "Base de conhecimento apagada com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
