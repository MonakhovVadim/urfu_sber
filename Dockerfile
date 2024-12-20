ARG PYTHON_VERSION=3.10.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN python3 -m venv venv

# activate venv
ENV PATH="/app/venv/bin:$PATH"

# Copy the source code into the container.
COPY . .

# create dir models, if not exists.
RUN test -e models || mkdir models

# set owner for data and models dirs
RUN chown appuser:appuser data models

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# run pipeline
RUN python src/data_generation.py
RUN python src/data_preprocessing.py
RUN python src/model_preparation.py
RUN python src/model_testing.py

# Expose the port that the application listens on.
EXPOSE 8501

# Run the application.
CMD streamlit run src/app.py
