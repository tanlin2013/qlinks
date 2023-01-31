FROM python:3.11-slim as python
ENV PYTHONUNBUFFERED=true
WORKDIR /app


FROM python as poetry
ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry config virtualenvs.in-project true
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.3.2
COPY . ./
RUN poetry install --no-interaction --no-ansi -vvv --no-dev


FROM python as runtime
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=poetry /app /app
ENTRYPOINT /bin/bash
