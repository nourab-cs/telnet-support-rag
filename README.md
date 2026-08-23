# TELNET Support Bot - Système RAG

Assistant conversationnel intelligent pour le support client TELNET SmartConnect, basé sur **RAG (Retrieval-Augmented Generation)**.

## 📦 Structure du projet

```
telnet-support-bot/
├── src/                           # Code source modulaire
│   ├── __init__.py
│   ├── document_loader.py         # Chargement multi-format de documents
│   ├── text_cleaner.py           # Nettoyage et prétraitement
│   ├── chunker.py                # Découpage intelligent
│   ├── embedder.py               # Génération d'embeddings
│   ├── chroma_store.py           # Base vectorielle ChromaDB
│   ├── retriever.py              # Récupération sémantique
│   ├── generator.py              # Génération de réponses
│   └── pipeline.py               # Pipeline RAG complet
├── data/                         # Documents source
│   ├── 01_guide_installation_smartconnect.md
│   ├── 02_faq_problemes_courants.md
│   ├── 03_documentation_api.md
│   ├── 04_guide_utilisateur_dashboard.md
│   ├── 05_procedure_maintenance.md
│   ├── 06_notes_version.md
│   ├── 07_configuration_reseau.md
│   └── 08_troubleshooting_guides.md
├── chroma_db/                    # Base vectorielle persistée
├── main.py                       # Point d'entrée principal
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation
```

## 🚀 Installation

### 1. Créer l'environnement virtuel

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Installer Ollama et le modèle

```bash
# Installation d'Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Télécharger le modèle Mistral
ollama pull mistral

# Lancer le serveur Ollama (dans un terminal séparé)
ollama serve
```

## � Utilisation

### Lancer le système

```bash
python main.py
```

### Questions d'exemple

- "Comment installer SmartConnect ?"
- "Quels sont les rôles utilisateurs disponibles ?"
- "Comment configurer MQTT ?"
- "Le service ne démarre pas, que faire ?"
- "Quelles sont les nouveautés de la version 3.3.0 ?"

## 🔧 Configuration

### Modifier les paramètres RAG

Les paramètres peuvent être ajustés dans les fichiers correspondants dans `src/` :

- **Chunking** : `src/chunker.py` - taille et chevauchement des chunks
- **Embeddings** : `src/embedder.py` - modèle d'embedding utilisé
- **Retrieval** : `src/retriever.py` - nombre de documents récupérés
- **LLM** : `src/generator.py` - modèle de langage et température

### Ajouter des documents

Placez vos fichiers dans le dossier `data/` :
- `.md` - Markdown
- `.txt` - Texte brut
- `.pdf` - PDF
- `.docx` - Word

## 🏗️ Architecture modulaire

Le système est organisé en modules indépendants :

1. **DocumentLoader** : Charge les documents de différents formats
2. **TextCleaner** : Nettoie et normalise le texte
3. **Chunker** : Découpe le texte en chunks cohérents
4. **Embedder** : Génère les embeddings vectoriels
5. **ChromaStore** : Gère la base vectorielle
6. **Retriever** : Récupère les documents pertinents
7. **Generator** : Génère les réponses avec le LLM
8. **Pipeline** : Orchestre l'ensemble du processus

## 📊 Performances

- **Embeddings** : BGE-M3 pour support multilingue optimal
- **Vector DB** : ChromaDB pour recherche sémantique rapide
- **LLM** : Mistral via Ollama pour réponses locales
- **Chunking** : Optimisé pour maximiser la pertinence

## 🐛 Dépannage

**Ollama connection refused** : Assurez-vous que `ollama serve` tourne dans un terminal séparé

**Mémoire insuffisante** : Utilisez un modèle plus léger comme `llama3.2:1b`

**Réponses incohérentes** : Vérifiez que vos documents couvrent bien le sujet des questions

## 📚 Extensions possibles

- Interface web avec Streamlit ou FastAPI
- Mode conversationnel avec mémoire
- Système multi-agent pour tâches complexes
- Évaluation automatique des performances
