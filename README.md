前端框架所用到的node_modules，下载链接如下：
链接：https://pan.baidu.com/s/1qUGbltG4GKnNJcfalYa13g?pwd=sljf 
提取码：sljf 

### 1.
将 `node_modules` ，放置 `Web` 目录下，即可在该目录按以下命令正常启动前端项目
```
npm run serve
```
### 2.
在 `Backend` 目录下，按以下命令安装后端依赖
```
pip install -r requirements.txt
```

### 3.
按照 `apikey.ini` 格式，设置后端程序所用api

### 4.
在Backend目录启动后端程序
```
uvicorn main:app --host 0.0.0.0 --port 8000
```
