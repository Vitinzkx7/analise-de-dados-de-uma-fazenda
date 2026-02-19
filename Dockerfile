# Dockerfile para Railway
FROM python:3.13-slim-bookworm

# Configurações de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de requisitos
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY . .

# Criar diretórios para outputs (serão montados como volumes no Railway)
RUN mkdir -p /app/outputs \
    "/app/Estatística descritiva" \
    "/app/Estatística inferencial" \
    "/app/Estatística bayesiana" \
    "/app/Modelos ML" \
    "/app/Modelos IA" \
    "/app/Deep Learning" \
    "/app/Comparacoes Modelos"

# Comando para executar a aplicação
CMD ["python", "main.py"]