from fastapi import Depends, FastAPI, HTTPException, File, Form, UploadFile, BackgroundTasks,HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from orm import crud, models, schemas, database
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from datetime import datetime, timedelta
import requests
import hashlib
import logging
import time
import random
import string
import os

from chat_paper import chat_paper_function, Reader, Paper
from utils import get_time_password, beautify_paper_output

from jose import JWTError, jwt
from typing import Union, Dict
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import openai
from pydantic import BaseModel
ChatModel = 1
get_db = 1

# 聊天接口
@app.post("/chat")
async def chat(chat_model: ChatModel, db: Session = Depends(get_db)):
    message_list = chat_model.message_list
    md5_hash = chat_model.md5_hash
    # 看文件是否在本地
    file = crud.get_paper_by_md5_hash(db, md5_hash)
    if file is None:
        raise HTTPException(status_code=400, detail="File not exists.")

    messages = [
        {"role": "system",
         "content": "You are a researcher in the field of [machine learning and deep learning] who is good at summarizing papers using concise statements and answering questions. You are powered by GPT-4."},
        {"role": "assistant",
         "content": "This the paper you need to summarize: " + file.content},
    ]

    message_list = eval(message_list)
    for msg in message_list:
        text = msg['data']['text']
        author = msg['author']
        print(msg)
        # Adding to messages with the role based on the author
        if author.lower() == 'gpt4':
            role = 'assistant'
        else:
            role = 'user'
            text += " Please use Chinese to answer questions."
        messages.append({"role": role, "content": text})

    # 用GPT分析论文
    reader = Reader(key_word='', query='', filter_keys='', sort='')
    openai.api_key = reader.chat_api_list[reader.cur_api]
    response = openai.ChatCompletion.create(
        model=reader.chatgpt_model,
        messages=messages
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    chat_history = schemas.ChatHistoryCreate(paper_id=file.id, prompt=message_list[-1]['data']['text'], response=result,
                                             created_at=datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    try:
        crud.create_chat_history(db=db, chat_history=chat_history)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    print('chat_history:', chat_history)

    return {'result': result}


def chat_summary(self, text, summary_prompt_token=1100):
    openai.api_key = self.chat_api_list[self.cur_api]
    text_token = len(self.encoding.encode(text))
    clip_text = text[:int(len(text) * (self.max_token_num - summary_prompt_token) / text_token)]
    messages = [
        {"role": "system",
         "content": "You are a researcher in the field of [" + self.key_word + "] who is good at summarizing papers using concise statements"},
        {"role": "assistant",
         "content": "This is the title, author, link, abstract and introduction of an English document. I need your help to read and summarize the following questions: " + clip_text},
        {"role": "user", "content": """                 
             1. Mark the title of the paper (with Chinese translation)
             2. list all the authors' names (use English)
             3. mark the first author's affiliation (output {} translation only)                 
             4. mark the keywords of this article (use English)
             5. link to the paper, Github code link (if available, fill in Github:None if not)
             6. summarize according to the following four points.Be sure to use {} answers (proper nouns need to be marked in English)
                - (1):What is the research background of this article?
                - (2):What are the past methods? What are the problems with them? Is the approach well motivated?
                - (3):What is the research methodology proposed in this paper?
                - (4):On what task and what performance is achieved by the methods in this paper? Can the performance support their goals?
             Follow the format of the output that follows:                  
             1. Title: xxx\n\n
             2. Authors: xxx\n\n
             3. Affiliation: xxx\n\n                 
             4. Keywords: xxx\n\n   
             5. Urls: xxx or xxx , xxx \n\n      
             6. Summary: \n\n
                - (1):xxx;\n 
                - (2):xxx;\n 
                - (3):xxx;\n  
                - (4):xxx.\n\n     

             Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.                 
             """.format(self.language, self.language, self.language)},
    ]

    response = openai.ChatCompletion.create(
            model=self.chatgpt_model,
            # prompt需要用英语替换，少占用token。
            messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result


async def get_summary_text(reader, paper, client_id):
    # 第一步先用title，abs，和introduction进行总结。
    text = ''
    text += 'Title:' + paper.title
    text += 'Url:' + paper.url
    text += 'Abstract:' + paper.abs
    text += 'Paper_info:' + paper.section_text_dict['paper_info']
    # intro
    text += list(paper.section_text_dict.values())[0]
    chat_summary_text = ""
    try:
        chat_summary_text = reader.chat_summary(text=text)
    except Exception as e:
        print("summary_error:", e)
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        if "maximum context" in str(e):
            current_tokens_index = str(e).find("your messages resulted in") + len(
                "your messages resulted in") + 1
            offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
            summary_prompt_token = offset + 1000 + 150
            chat_summary_text = reader.chat_summary(text=text, summary_prompt_token=summary_prompt_token)

    return chat_summary_text


def download_pdf(db, url):  # 接收数据库会话 db 和文件的 url 作为参数
    response = requests.get(url)
    if response.headers['Content-Type'] != 'application/pdf':  # 检查响应头中的 Content-Type 是否为 application/pdf，确保下载的是 PDF 文件
        raise ValueError("URL does not contain a PDF file.")

    # 使用MD5哈希值作为文件名 如果文件已经存在则直接返回文件路径
    file_md5 = hashlib.md5(response.content).hexdigest()
    file = crud.get_paper_by_md5_hash(db, file_md5)

    if file is not None:
        return file.path

    # 保存文件到本地
    static_dir = 'static'
    pdf_dir = 'pdf'
    year, month, day = datetime.now().strftime('%Y/%m/%d').split('/')
    base_dir = os.path.join(static_dir, pdf_dir, year, month, day)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    file_name = file_md5 + '.pdf'
    file_path = os.path.join(base_dir, file_name)
    with open(file_path, 'wb') as f:
        f.write(response.content)

    return file_path