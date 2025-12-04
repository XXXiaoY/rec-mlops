# 项目快速启动指南

以下命令请**严格按顺序逐条复制执行**（已测试可直接运行）。

## 操作步骤

1. **启动所有服务**

    ```bash
    # 1. 启动所有服务
    docker-compose up -d
    ```

2. **获取 spark-master 容器 ID**

    ```bash
    # 2. 获取 spark-master 容器 ID
    CONTAINER=$(docker-compose ps -q spark-master)
    ```

3. **在容器内安装基础依赖**

    ```bash
    # 3. 在容器内安装基础依赖
    docker exec -it $CONTAINER pip install \
        prefect \
        mlflow \
        structlog \
        pandas \
        scikit-learn \
        pyyaml \
        delta-spark==2.4.0 \
        kafka-python==2.0.2
    ```

4. **安装 Prefect 2.x 版本（必须 <3.0）**

    ```bash
    # 4. 安装 Prefect 2.x 版本（必须 <3.0）
    docker exec -it $CONTAINER pip install "prefect>=2.0.0,<3.0.0"
    ```

5. **初始化 Delta Lake 表结构**

    ```bash
    # 5. 初始化 Delta Lake 表结构（在 MinIO 中创建所需表）
    docker exec -w /app -it $CONTAINER python src/init_delta_tables.py
    ```

6. **运行模型重训练 Flow（关键步骤）**

    ```bash
    # 6. 运行模型重训练 Flow（关键步骤）
    docker exec \
      -e PREFECT_HOME=/tmp/.prefect \
      -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
      -e PREFECT_API_URL=http://prefect:4200/api \
      -w /app \
      -it $CONTAINER python -m src.pipelines.retraining_flow
    ```

## 预期结果

- Prefect UI（http://localhost:4200）显示 Flow 运行成功（绿色）
- MLflow UI（http://localhost:5000） → Models 页面出现注册模型：Recommendation_SVD，最新版本被标记为 Production

## 模型热加载验证

```bash
Bash# 7. 测试模型热加载 API（验证 Production 模型是否成功加载到推荐服务）
docker exec -it $CONTAINER python -c \
  "import requests; print(requests.post('http://localhost:8000/admin/reload-models').json())"
```

预期返回（示例）：

```json
JSON{
  "status": "success",
  "message": "Models reloaded from Production",
  "current_state": {
    "Recommendation_SVD": {
      "version": "1",
      "stage": "Production",
      "loaded_at": "2025-12-04T08:88:88.888888"
    }
  }
}
```