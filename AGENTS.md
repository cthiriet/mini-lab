# mini-lab

- `~/VM/mini-lab/.venv` est le venv du Mac : ne jamais le créer, le reconstruire ni le supprimer depuis la VM.
- Dans la VM, préfixer chaque commande uv : `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/mini-lab nice -n 10 uv run pytest -q` (un `uv run` nu recréerait un `.venv` Linux dans le dossier partagé).
- Pas de MPS ni de GPU dans la VM : l'entraînement lourd se lance sur le Mac.
