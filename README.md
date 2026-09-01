# TELNET Support Bot - Système RAG

Assistant conversationnel intelligent pour le support client TELNET SmartConnect, basé sur **RAG (Retrieval-Augmented Generation)**.

## 📦 Structure du projet

```
telnet-support-bot/
├── src/                           # Code source modulaire
│   ├── __init__.py
│   ├── document_loader.py         # Chargement multi-format de documents
│   ├── text_cleaner.py           # Nettoyage et prétraitement
│   ├── chunking_agent.py         # Agent IA de chunking intelligent
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
├── agent_chunks.py               # Script de test de l'agent de chunking
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

## 💬 Utilisation

### Lancer le système principal

```bash
python main.py
```

### Tester l'agent de chunking

```bash
python agent_chunks.py
```

Ce script teste l'agent IA de chunking avec vos documents et affiche les statistiques de chunking.

### Questions d'exemple

- "Comment installer SmartConnect ?"
- "Quels sont les rôles utilisateurs disponibles ?"
- "Comment configurer MQTT ?"
- "Le service ne démarre pas, que faire ?"
- "Quelles sont les nouveautés de la version 3.3.0 ?"

### Commandes conversationnelles

Le système supporte maintenant les commandes spéciales pendant la conversation :

- `quit` : quitter l'application
- `history` : afficher l'historique de conversation
- `clear` : effacer l'historique
- `summary` : afficher un résumé de la conversation

### Tester le système conversationnel

```bash
python test_conversation.py
```

Ce script teste automatiquement les fonctionnalités de conversation avec historique.

## 🤖 Agent IA de Chunking

Le système utilise un **agent IA de chunking intelligent** qui analyse automatiquement chaque document pour déterminer les meilleurs paramètres de découpage.

### Fonctionnalités

- **Analyse automatique** : Le LLM analyse la structure et le contenu de chaque document
- **Adaptation dynamique** : Chunk size et overlap ajustés selon le type de contenu
- **Détection de structure** : Identifie les titres, listes, et sections
- **Stratégies multiples** : Standard, agressive, ou conservatrice selon le document
- **Métadonnées enrichies** : Chaque chunk contient ses paramètres de chunking

### Résultats typiques

Avec l'agent de chunking :
- **72 chunks créés** à partir de 8 documents
- **Taille moyenne** : 350 caractères
- **Stratégie adaptative** : Standard pour la plupart, agressive pour les documents structurés
- **Chevauchement intelligent** : 88 caractères en moyenne pour maintenir le contexte

## � Historique de Conversation

Le système supporte maintenant le **mode conversationnel avec mémoire** pour maintenir le contexte des échanges.

### Fonctionnalités

- **Mémoire conversationnelle** : Conserve jusqu'à 10 échanges question-réponse
- **Contexte intelligent** : L'historique est inclus dans les prompts pour des réponses contextuelles
- **Commandes spéciales** : `history`, `clear`, `summary` pour gérer la conversation
- **Multi-sessions** : Support pour plusieurs sessions de conversation indépendantes

### Avantages

- Meilleure compréhension des questions de suivi
- Réponses plus cohérentes dans le temps
- Possibilité de se référer aux échanges précédents
- Gestion flexible de la mémoire de conversation

## �🔧 Configuration

### Modifier les paramètres RAG

Les paramètres peuvent être ajustés dans les fichiers correspondants dans `src/` :

- **Chunking Agent** : `src/chunking_agent.py` - agent IA et stratégie de chunking
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
3. **ChunkingAgent** : Agent IA pour chunking intelligent et adaptatif
4. **Embedder** : Génère les embeddings vectoriels
5. **ChromaStore** : Gère la base vectorielle
6. **Retriever** : Récupère les documents pertinents
7. **Generator** : Génère les réponses avec le LLM
8. **Pipeline** : Orchestre l'ensemble du processus

## 📊 Performances

- **Embeddings** : BGE-M3 pour support multilingue optimal
- **Vector DB** : ChromaDB pour recherche sémantique rapide
- **LLM** : Mistral via Ollama pour réponses locales
- **Chunking** : Agent IA adaptatif pour optimisation automatique

## 🐛 Dépannage

**Ollama connection refused** : Assurez-vous que `ollama serve` tourne dans un terminal séparé

**Mémoire insuffisante** : Utilisez un modèle plus léger comme `llama3.2:1b`

**Réponses incohérentes** : Vérifiez que vos documents couvrent bien le sujet des questions

**Chunking lent** : L'agent IA analyse chaque document, cela peut prendre du temps. Pour du chunking statique rapide, remplacez `ChunkingAgent` par `Chunker` dans `pipeline.py`.

## 📚 Extensions possibles

- Interface web avec Streamlit ou FastAPI
- Mode conversationnel avec mémoire
- Système multi-agent pour tâches complexes
- Évaluation automatique des performances
- Dashboard d'administration
