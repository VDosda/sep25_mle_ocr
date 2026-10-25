### STAGE 1 : BUILDER
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    nodejs npm git curl libgl1 libglib2.0-0 build-essential \
    && apt-get clean

RUN pip install --upgrade pip

# Copier requirements-jupyter.txt dans le builder
COPY requirements-jupyter.txt /tmp/requirements.txt

# Installer les dépendances Python du projet
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Build JupyterLab assets (RSPack, no jlpm)
RUN jupyter lab build --dev-build=False --minimize=False

### STAGE 2 : FINAL RUNTIME
FROM python:3.10-slim

RUN useradd -m appuser

RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 && apt-get clean
RUN pip install --upgrade pip

# Copier requirements-jupyter.txt dans le runtime
COPY requirements-jupyter.txt /tmp/requirements.txt

# Installer les dépendances Python du runtime
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copier les assets JupyterLab compilés depuis le builder
COPY --from=builder /usr/local/share/jupyter/lab /usr/local/share/jupyter/lab

WORKDIR /workspace
RUN chown -R appuser:appuser /workspace

# Ajouter la configuration Jupyter
RUN mkdir -p /home/appuser/.jupyter
COPY jupyter_server_config.py /home/appuser/.jupyter/jupyter_server_config.py
RUN chown -R appuser:appuser /home/appuser/.jupyter

USER appuser

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password=", "--NotebookApp.notebook_dir=/workspace"]
