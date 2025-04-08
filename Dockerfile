FROM python:3.12

WORKDIR /app

COPY server.py model.pkl tfidf.pkl svd.pkl requirements.txt ./

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# RUN python -c "import nltk; nltk.download('all')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
