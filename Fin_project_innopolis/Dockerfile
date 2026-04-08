# Указываем базовый образ с нужной версией Python.
# slim-версия — это компромисс между размером и наличием необходимых утилит.
FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера.
# Все последующие команды будут выполняться из этой папки.
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их.
# Сначала копируем только requirements.txt, чтобы Docker использовал кеш.
COPY requirements.txt .
RUN apt-get update && apt-get install -y
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код проекта в рабочую директорию.
COPY . .

# Команда для запуска приложения при старте контейнера.
CMD ["streamlit", "run", "main.py", "--server.port=8000", "--server.address=0.0.0.0"]
