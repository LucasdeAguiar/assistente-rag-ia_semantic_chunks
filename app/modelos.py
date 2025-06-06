from pydantic import BaseModel

class TextoEntrada(BaseModel):
    texto: str

class PerguntaEntrada(BaseModel):
    pergunta: str
