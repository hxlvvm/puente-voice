# The phone agent for the demo deployment (compose.yaml). CPU by default; models are mounted, not baked in.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY puente_voice puente_voice
COPY config config
# editable: the demo clinic is read from /app/config next to the package
RUN pip install --no-cache-dir -e ".[audio]"
CMD ["python", "-m", "puente_voice.serve"]
