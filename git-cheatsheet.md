# Venv & Git – schnelle Problemlösungen (adhoc-2025)

## Venv einrichten
python3 -m venv .venv  
source .venv/bin/activate
pip install -U pip 
pip install pandas numpy matplotlib jupyter ipykernel
pip list
pip freeze > requirements.txt
which python

## Git initiieren
git init
.gitignore anlegen (siehe unten)
git status (untracked files werden angezeigt)
git add . (alles hinzufügen, ersten commit vorbereiten)
git commit -m "Initial commit" (projekt jetzt lokal versioniert)

Auf github.com:
new repository
name wählen, kein readme, kein .gitignore, keine licence
create repository 
github zeigt danach remote-url: git@github.com:username/repo.git

git remote add origin git@github.com:USERNAME/REPO.git (remote hinzufügen)
git remote -v (prüfen)
git branch -M main (branch korrekt setzen)
git push -u origin main (erster push)

# HF Spaces als zweites Remote hinzufügen
unter huggingface.co/new-space new space anlegen 
unter huggingface.co/settings/tokens new token factagent-deploy anlegen

git add .
git commit -m "Add HF Spaces deployment"
git tag -a v6.0 -m "Deployment-ready for Hugging Face Spaces"

git remote add hf https://huggingface.co/spaces/thiev980/factagent
git remote set-url hf https://thiev980:hf_DEIN_TOKEN@huggingface.co/spaces/thiev980/factagent
git push hf main (allenfalls git push hf main --force)

## Tags
git tag -a v5.0 -m "Feature-complete: Streaming, HITL, History DB, Source Graph"
git push origin v5.0
git checkout v5.0 --> zu diesem Stand zurückkehren

## Push rejected / divergent branches
git pull --rebase origin master
git status (solve if necessary)
git push

## Grosses CSV korrekt via LFS
brew install git-lfs
git lfs install

git lfs track "<path>"
git add .gitattributes
git commit -m "track via lfs"
git add <path>
git commit -m "add csv"
git push

## Push blockiert wegen >100MB
git reset --soft HEAD~1
git rm --cached <bigfile>
git commit -m "commit without big file"
git push

## Aus Versehen commited
git reset HEAD~1 (aus staging entfernen)

## Minimales .gitignore
    # Python / venv
    .venv/
    __pycache__/
    *.pyc

    # Jupyter
    .ipynb_checkpoints/

    # macOS
    .DS_Store

    # Editor
    .vscode/

    # Env
    .env