#!/bin/bash

cd src

# Detectar sistema operativo
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  # Windows
  if [ ! -d "venv" ]; then
    echo "Creando entorno virtual en Windows..."
    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt
  else
    source venv/Scripts/activate
    echo "Entorno virtual ya existente en Windows."
  fi
else
  # Ubuntu u otros sistemas basados en Unix
  if [ ! -d "venv" ]; then
    echo "Creando entorno virtual en Ubuntu/MacOS..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
  else
    source venv/bin/activate
    echo "Entorno virtual ya existente en Ubuntu."
  fi
fi

# Ejecutar el programa
python __main__.py
