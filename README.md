# README

为了防止用户执行 `npm install`指令之后缺少节点的问题，这里提供前端框架所用到的`node_modules`。下载链接如下：

链接：https://pan.baidu.com/s/1qUGbltG4GKnNJcfalYa13g?pwd=sljf 
提取码：sljf 

### 1.
将 `node_modules` ，放置 `Web` 目录下，即可在该目录按照以下命令正常启动前端项目
```
npm run serve
```
### 2.
在 `Backend` 目录下，按以下命令安装后端依赖
```
pip install -r requirements.txt
```

### 3.
按照 `apikey.ini` 格式，设置后端程序所用 `apikey`

### 4.
在`Backend`目录启动后端程序命令
```
uvicorn main:app --host 0.0.0.0 --port 8000
```
