FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the FastAPI application
# CMD ["fastapi", "run", "./msel_chat_langchain.py"]
CMD ["fastapi", "run", "./msel_chat_langgraph.py"]