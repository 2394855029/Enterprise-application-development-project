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

BASE_URL = "http://localhost:8000/"

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

if not os.path.exists('static'):
    os.makedirs('static')
app.mount("/static", StaticFiles(directory="static"), name="static") # 将 static 目录挂载到 FastAPI 应用上，使得该目录下的静态文件可以通过 /static 路径访问

origins = [
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logfile_path = os.path.join('log', 'server.log')
if not os.path.exists('log'):
    os.makedirs('log')
fh = logging.FileHandler(filename='log/server.log', encoding='utf-8')
# 按照日期切割日志文件
fh = logging.handlers.TimedRotatingFileHandler(filename='log/server.log', when='D', interval=1, backupCount=7, encoding='utf-8')
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch) #将日志输出至屏幕
logger.addHandler(fh) #将日志输出至文件

logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_requests(request, call_next):
    idem = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    logger.info(f"rid={idem} completed_in={formatted_process_time}ms status_code={response.status_code}")
    
    return response

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


def download_pdf(db, url):# 接收数据库会话 db 和文件的 url 作为参数
    response = requests.get(url)
    if response.headers['Content-Type'] != 'application/pdf':# 检查响应头中的 Content-Type 是否为 application/pdf，确保下载的是 PDF 文件
        raise ValueError("URL does not contain a PDF file.")
    
    logger.info(f"response.status_code={response.status_code}")
    # 使用MD5哈希值作为文件名 如果文件已经存在则直接返回文件路径
    file_md5 = hashlib.md5(response.content).hexdigest()
    file = crud.get_paper_by_md5_hash(db, file_md5)
    logger.info(f"file={file}")
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


# 接收PDF的URL，下载PDF返回文件url
@app.get("/download")
def upload(pdf_url: str, db: Session = Depends(get_db)):
    logger.info(f"pdf_url={pdf_url}")
    if pdf_url.startswith('static'):
        return {'url': BASE_URL + pdf_url}
    if not pdf_url:
        raise HTTPException(status_code=400, detail="No PDF URL provided.")
    try:
        file_path = download_pdf(db, pdf_url)
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"file_path={file_path}")
    # 将文件路径转换成url，windows和linux的路径分隔符不一样
    file_path = file_path.replace('\\', '/')
    file_url = BASE_URL + file_path
    return {'url': file_url}


# 获取全部论文
@app.get("/papers", response_model=list[schemas.Paper])
def read_papers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    papers = crud.get_papers(db, skip=skip, limit=limit)
    return papers

# 将post表单内的论文存到本地并返回文件url
@app.post("/upload")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"file={file}")
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")
    try:
        # 保存文件到本地
        static_dir = 'static'
        pdf_dir = 'pdf'
        year, month, day = datetime.now().strftime('%Y/%m/%d').split('/')
        base_dir = os.path.join(static_dir, pdf_dir, year, month, day)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # file_name = file.filename
        # 文件名为文件的md5值
        
        # 读取文件内容
        # 读取文件内容
        file_content = await file.read()

        # 计算文件的MD5值
        md5_hash = hashlib.md5()
        md5_hash.update(file_content)
        file_name = md5_hash.hexdigest() + '.pdf'

        # 保存文件到本地
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(file_content)

        # 文件处理完成后，记得关闭文件
        await file.close()

    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"file_path={file_path}")
    # 将文件路径转换成url
    file_path = file_path.replace('\\', '/')
    file_url = BASE_URL + file_path
    return {'url': file_url}

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_text(message)

    async def broadcast(self, message: str):
        for websocket in self.active_connections.values():
            await websocket.send_text(message)

manager = ConnectionManager()

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
    
    # logger.info()
    # 发送到前端
    # await send_message_to_client(client_id, chat_summary_text)
    return chat_summary_text

async def get_method_text(reader, paper, chat_summary_text, client_id):
    # 第二步总结方法：
    # TODO，由于有些文章的方法章节名是算法名，所以简单的通过关键词来筛选，很难获取，后面需要用其他的方案去优化。
    method_key = ''
    for parse_key in paper.section_text_dict.keys():
        if 'method' in parse_key.lower() or 'approach' in parse_key.lower():
            method_key = parse_key
            break

    if method_key != '':
        text = ''
        method_text = ''
        summary_text = ''
        summary_text += "<summary>" + chat_summary_text
        # methods                
        method_text += paper.section_text_dict[method_key]
        text = summary_text + "\n\n<Methods>:\n\n" + method_text
        chat_method_text = ""
        try:
            chat_method_text = reader.chat_method(text=text)
        except Exception as e:
            print("method_error:", e)
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            if "maximum context" in str(e):
                current_tokens_index = str(e).find("your messages resulted in") + len(
                    "your messages resulted in") + 1
                offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                method_prompt_token = offset + 800 + 150
                chat_method_text = reader.chat_method(text=text, method_prompt_token=method_prompt_token)
        # await send_message_to_client(client_id, chat_method_text)
    else:
        chat_method_text = ''
    
    return chat_method_text


async def get_conclusion_text(reader, paper, chat_summary_text, chat_method_text, client_id):
    # 第三步总结全文，并打分：
    conclusion_key = ''
    for parse_key in paper.section_text_dict.keys():
        if 'conclu' in parse_key.lower():
            conclusion_key = parse_key
            break

    text = ''
    conclusion_text = ''
    summary_text = ''
    summary_text += "<summary>" + chat_summary_text + "\n <Method summary>:\n" + chat_method_text
    if conclusion_key != '':
        # conclusion                
        conclusion_text += paper.section_text_dict[conclusion_key]
        text = summary_text + "\n\n<Conclusion>:\n\n" + conclusion_text
    else:
        text = summary_text
    chat_conclusion_text = ""
    try:
        chat_conclusion_text = reader.chat_conclusion(text=text)
    except Exception as e:
        print("conclusion_error:", e)
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        if "maximum context" in str(e):
            current_tokens_index = str(e).find("your messages resulted in") + len(
                "your messages resulted in") + 1
            offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
            conclusion_prompt_token = offset + 800 + 150
            chat_conclusion_text = reader.chat_conclusion(text=text,
                                                        conclusion_prompt_token=conclusion_prompt_token)
    # await send_message_to_client(client_id, chat_conclusion_text)
    return chat_conclusion_text


async def analyze_pdf(db, file_path, client_id):
    # 用GPT分析论文
    logger.info(f"file_path={file_path}")
    htmls = []
    try:
        reader = Reader(key_word='', query='', filter_keys='', sort='')
        paper = Paper(path=file_path)
        
        chat_summary_text = await get_summary_text(reader, paper, client_id)

        htmls.append(chat_summary_text)

        chat_method_text = await get_method_text(reader, paper, chat_summary_text, client_id)

        htmls.append(chat_method_text)

        chat_conclusion_text = await get_conclusion_text(reader, paper, chat_summary_text, chat_method_text, client_id)

        htmls.append(chat_conclusion_text)
        
        result = "\n".join(htmls)

    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    content = '\n'.join(paper.section_text_dict.values())
    # logger.info(f"chat_summary_text={chat_summary_text}")
    title, authors, keywords = beautify_paper_output(result)
    logger.info(f"title={title}")

    # md5是文件名 提取出
    file_md5 = file_path.split('/')[-1].split('.')[0]
    
    # 当前时间转换为字符串
    date_str = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    p1 = crud.get_paper_by_md5_hash(db, file_md5)
    if p1 is None:
        # 将分析结果保存到数据库
        paper = schemas.PaperCreate(title=title, abstract=None, authors=authors, keywords=keywords, content=content, analysis_result=result, created_at=date_str, updated_at=date_str, md5_hash=file_md5, path=file_path)
        try:
            crud.create_paper(db=db, paper=paper)
        except Exception as e:
            logger.error(f"Exception: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # 更新分析结果
        paper = schemas.PaperCreate(title=title, abstract=None, authors=authors, keywords=keywords, content=content, analysis_result=result, updated_at=date_str)
        try:
            crud.update_paper(db=db, paper_id=p1.id, paper=paper)
        except Exception as e:
            logger.error(f"Exception: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    return result

# 用GPT分析论文, 因为十分耗时，所以用异步函数提交后台处理
# @app.get("/analyze")
# async def analyze(file_path: str, client_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
#     logger.info(f"file_path={file_path}")
#     if not file_path:
#         raise HTTPException(status_code=400, detail="No file provided.")
#     # 看文件是否在本地
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=400, detail="File not exists.")
#     # 如果文件已经分析过，则直接返回
#     file_md5 = file_path.split('/')[-1].split('.')[0]
#     file = crud.get_paper_by_md5_hash(db, file_md5)
#     logger.info(f"file={file}")
#     if file is not None:
#         return {'result': file.analysis_result}
#     try:
#         # 用GPT分析论文
#         background_tasks.add_task(analyze_pdf, db, file_path, client_id)
#     except Exception as e:
#         logger.error(f"Exception: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

#     return {'status': 'success'}

@app.get("/analyze")
async def analyze(file_path: str, client_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    logger.info(f"file_path={file_path}")
    if not file_path:
        raise HTTPException(status_code=400, detail="No file provided.")
    # 看文件是否在本地，将文件url转换成路径
    path_parts = file_path.split('/')
    if not os.path.exists(os.path.join(*path_parts)):
        raise HTTPException(status_code=400, detail="File not exists.")
    # 如果文件已经分析过，则直接返回
    file_name = path_parts[-1]
    file_md5 = file_name.split('.')[0]
    file = crud.get_paper_by_md5_hash(db, file_md5)
    logger.info(f"file={file}")
    if file is not None:
        return {'result': file.analysis_result}
    try:
        # 用GPT分析论文
        result = await analyze_pdf(db, file_path, client_id)
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    return {'result': result}

# 获取最近分析的论文
@app.get("/recent_papers", response_model=list[schemas.Paper])
def read_recent_papers(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    papers = crud.get_recent_papers(db, skip=skip, limit=limit)
    return papers

SECRET_KEY = "09d25e094faa6ca2556c92sc52b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.get("/login")
async def login_for_access_token(password: str):
    logger.info(f"password={password}")
    time_password = get_time_password()
    if password != time_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": 'guest'}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            print('data: ', data)
            # 处理接收到的数据
    except WebSocketDisconnect:
        print('WebSocketDisconnect')
        manager.disconnect(client_id)

@app.post("/send/{client_id}")
async def send_message_to_client(client_id: str, message: str):
    await manager.send_personal_message(message, client_id)
    return {"message": "Message sent to client"}

class ChatModel(BaseModel):
    message_list: str
    md5_hash: str

# 聊天接口
@app.post("/chat")
async def chat(chat_model: ChatModel, db: Session = Depends(get_db)):
    message_list = chat_model.message_list
    md5_hash = chat_model.md5_hash
    logger.info(f"message_list={message_list}")
    # 看文件是否在本地
    file = crud.get_paper_by_md5_hash(db, md5_hash)
    logger.info(f"file={file}")
    if file is None:
        raise HTTPException(status_code=400, detail="File not exists.")
    
    messages = [
        {"role": "system",
        "content": "You are a researcher in the field of [machine learning and deep learning] who is good at summarizing papers using concise statements and answering questions. You are powered by GPT-4."},
        {"role": "assistant",
        "content": "This the paper you need to summarize: " + file.content},
        # {"role": "user",
        # "content": "Here is my question: " + message_list[0] + " Please use Chinese to ask questions."},
    ]

    message_list = eval(message_list)
    for msg in message_list:
        text = msg['data']['text']
        author = msg['author']
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
    print("result:\n", result)
    print('chat_summary')
    print("prompt_token_used:", response.usage.prompt_tokens,
            "completion_token_used:", response.usage.completion_tokens,
            "total_token_used:", response.usage.total_tokens)

    chat_history = schemas.ChatHistoryCreate(paper_id=file.id, prompt=message_list[-1]['data']['text'], response=result,
                                             created_at=datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    try:
        crud.create_chat_history(db=db, chat_history=chat_history)
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    print('chat_history:', chat_history)

    return {'result': result}

# 删除论文
@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: int, db: Session = Depends(get_db)):
    db_paper = crud.delete_paper(db=db, paper_id=paper_id)
    return db_paper

# 论文详情
@app.get("/papers/{paper_id}", response_model=schemas.Paper)
def read_paper(paper_id: int, db: Session = Depends(get_db)):
    db_paper = crud.get_paper(db, paper_id=paper_id)
    if db_paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return db_paper

# 更新论文
@app.put("/papers/{paper_id}", response_model=schemas.Paper)
def update_paper(paper_id: int, paper: schemas.PaperCreate, db: Session = Depends(get_db)):
    db_paper = crud.update_paper(db=db, paper_id=paper_id, paper=paper)
    return db_paper

# 重新分析论文
@app.get("/re_analyze")
async def re_analyze_paper(md5_hash: str, db: Session = Depends(get_db)):
    file = crud.get_paper_by_md5_hash(db, md5_hash)
    if file is None:
        raise HTTPException(status_code=400, detail="File not exists.")
    file_path = file.path
    try:
        # 用GPT分析论文
        result = await analyze_pdf(db, file_path, 'client_id')
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    return {'result': result}