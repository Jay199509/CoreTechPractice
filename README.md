# AI Agent相关练习

## 运行指南

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API客户端

代码中均使用OpenAI兼容格式的API，只需要在环境变量中配置即可

```bash
client = OpenAI(
        api_key=os.getenv("AGICTO_API_KEY"),
        base_url=os.getenv("AGICTO_BASE_URL")
    )
```

### 3.运行程序

```
python <appname>.py
```

## 浏览器兼容性

推荐使用 Chrome、Firefox、Edge 等现代浏览器访问。
