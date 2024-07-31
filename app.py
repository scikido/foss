from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from brain import *
import os

app = FastAPI()


@app.post("/suggestions/")
async def process_text(file: UploadFile = File(...)):
    if file is not None:
        content = await file.read()

        text = content.decode('utf-8')
        print("text extracted!")

        retriever, llm = processing_embedding(text)
        print("Vectorstore and LLM created!")
        sugg_prt = "Based on the sentiment and topics, generate a list of 5 actionable areas for improvement in service. Provide detailed suggestions and examples for each area."
        response = answer_question(retriever, llm, question=sugg_prt)
        result = str(response)

        return JSONResponse(content={"generated_text": result})
    
@app.post("/issue/")
async def process_text(file: UploadFile = File(...)):
    if file is not None:
        content = await file.read()
        text = content.decode('utf-8')
        print("text extracted!")

        retriever, llm = processing_embedding(text)
        print("Vectorstore and LLM created!")
        issue_prt = "Top 5 issues faced by customers that need to be fixed urgently. Give me the issue and then type of priority(high, low, medium) also suggest method to improve it"
        response = answer_question(retriever, llm, question=issue_prt)
        result = str(response)

        return JSONResponse(content={"generated_text": result})