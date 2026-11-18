# BiliMind：B 站收藏视频知识库


## 产品定位


## 系统架构



## 当前流程


## 功能模块


## 项目亮点


## 限制与下一步可优化方向


## 本地运行

1. 安装 Python 依赖：`pip install -r requirements.txt`。复制配置：PowerShell 中执行 `Copy-Item .env.example .env`，填写 `DASHSCOPE_API_KEY`。更换 Embedding 模型后需要重建向量索引；本地音频 ASR 兜底依赖 `ffmpeg`。
2. 在项目根目录启动后端：`python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload`。接口文档位于 `http://127.0.0.1:8001/docs`。
3. 进入 `frontend`，执行 `npm ci`、`npm run dev`，打开 `http://localhost:3000`。若后端地址不同，设置 `NEXT_PUBLIC_API_URL` 并重启前端。
4. 扫码登录，选择一个小收藏夹入库。完成后试问找回问题，并在笔记问题中写明已入库视频的 BV 号；通过来源链接核对结果。