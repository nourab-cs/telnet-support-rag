# TELNET Support Bot - RAG System

Un système de support technique intelligent basé sur RAG (Retrieval-Augmented Generation) pour TELNET SmartConnect.

## 🌟 Fonctionnalités

- **Recherche sémantique avancée** : Utilise des embeddings BGE-M3 pour une recherche précise dans la documentation
- **Chunking intelligent** : Agent IA adaptatif qui optimise le découpage des documents selon leur structure
- **Génération de réponses strictes** : LLM configuré pour répondre uniquement avec les informations de la documentation
- **Historique conversationnel** : Supporte le contexte de conversation pour des interactions naturelles
- **Base vectorielle persistante** : ChromaDB pour un stockage et une récupération rapides
- **Validation de contenu** : Détection automatique de réponses suspectes ou hors sujet

## 📋 Prérequis

- Python 3.8+
- Ollama (avec modèle Mistral installé)
- 4GB+ RAM

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <repository-url>
cd telnet-support-bot
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Installer et configurer Ollama

```bash
# Installer Ollama (si non déjà installé)
# Visiter https://ollama.ai pour les instructions

# Télécharger le modèle Mistral
ollama pull mistral

# Démarrer le serveur Ollama
ollama serve
```

### 4. Préparer les documents

Placez vos fichiers de documentation Markdown dans le dossier `data/` :

```
data/
├── 01_guide_installation_smartconnect.md
├── 02_faq_problemes_courants.md
├── 03_documentation_api.md
├── 04_guide_utilisateur_dashboard.md
├── 05_procedure_maintenance.md
├── 06_notes_version.md
├── 07_configuration_reseau.md
└── 08_troubleshooting_guide.md
```

## 🎯 Utilisation

### Lancer le bot

```bash
python main.py
```

### Commandes disponibles

Une fois le bot lancé, vous pouvez utiliser les commandes suivantes :

- **Questions posées** : Posez vos questions sur TELNET SmartConnect
- **`quit`** : Quitter l'application
- **`history`** : Afficher l'historique de conversation
- **`clear`** : Effacer l'historique
- **`summary`** : Obtenir un résumé de la conversation

### Exemples de questions

```
Comment lister les devices ?
Quels sont les ports de configuration ?
Comment résoudre les problèmes de connexion ?
Quelle est la procédure de maintenance ?
```

## 🏗️ Architecture

```
telnet-support-bot/
├── main.py                 # Point d'entrée principal
├── data/                   # Documents de documentation
├── chroma_db/              # Base vectorielle (générée automatiquement)
├── src/
│   └── rag/
│       ├── __init__.py              # Exports du package
│       ├── pipeline.py              # Pipeline RAG principal
│       ├── document_loader.py       # Chargement des documents
│       ├── chunking_agent.py        # Agent de chunking intelligent
│       ├── embedder.py              # Génération d'embeddings
│       ├── chroma_store.py          # Gestion ChromaDB
│       ├── retriever.py             # Récupération sémantique
│       ├── generator.py             # Génération de réponses
│       └── conversation_history.py   # Gestion de l'historique
└── requirements.txt        # Dépendances Python
```

## 🔧 Configuration

Les paramètres par défaut sont configurés dans `src/rag/pipeline.py` via la classe `RAGConfig` :

```python
class RAGConfig:
    DATA_DIR = "./data"
    DB_DIR = "./chroma_db"
    COLLECTION_NAME = "telnet_support"
    USE_HISTORY = True
    MAX_HISTORY = 10
    ENABLE_LLM_ANALYSIS = True
```

## 🧠 Pipeline RAG

Le système suit ce pipeline :

1. **Chargement des documents** : Lecture et nettoyage des fichiers Markdown
2. **Chunking intelligent** : Découpage adaptatif par agent IA
3. **Embeddings** : Génération de vecteurs avec BGE-M3
4. **Indexation** : Stockage dans ChromaDB
5. **Récupération** : Recherche sémantique des documents pertinents
6. **Génération** : Création de réponses basées sur le contexte
7. **Validation** : Filtrage des réponses suspectes

## 🛠️ Composants

### DocumentLoader
Charge et nettoie les documents Markdown avec un prétraitement léger.

### ChunkingAgent
Agent IA qui analyse la structure des documents et optimise les paramètres de chunking :
- Détection automatique de la structure (titres, listes, tableaux)
- Analyse LLM ou heuristique selon la configuration
- Filtrage des chunks trop petits

### Embedder
Gère le modèle d'embeddings BGE-M3 pour la vectorisation sémantique.

### ChromaStore
Interface avec ChromaDB pour le stockage et la récupération vectorielle.

### Retriever
Effectue la recherche sémantique avec plusieurs stratégies :
- `similarity` : Recherche par similarité standard
- `mmr` : Maximal Marginal Relevance (diversité)
- `similarity_score_threshold` : Filtrage par seuil

### Generator
Génère des réponses avec des prompts stricts pour éviter les hallucinations :
- Répond uniquement avec le contexte fourni
- Détection de contenu suspect
- Validation automatique des réponses

### ConversationHistory
Gère l'historique de conversation pour le contexte multi-tours.

## 🔍 Dépannage

### Problème : Import error pour les modules

**Solution** : Assurez-vous que le dossier `src` est dans le PYTHONPATH ou utilisez le script `main.py` qui configure le chemin automatiquement.

### Problème : Ollama ne répond pas

**Solution** : Vérifiez que le serveur Ollama est en cours d'exécution :
```bash
ollama serve
```

### Problème : Réponses hors sujet

**Solution** : Le système inclut une validation automatique. Si le problème persiste, vérifiez :
- La qualité des documents dans `data/`
- Les paramètres de température du LLM (doit être bas, ~0.0)
- Le seuil de similarité du retriever

### Problème : Base vectorielle corrompue

**Solution** : Supprimez le dossier `chroma_db/` et relancez le bot pour reconstruire l'index.

## 📊 Performance

- **Temps de première exécution** : ~2-3 minutes (téléchargement du modèle BGE-M3)
- **Temps de réponse** : ~1-2 secondes par question
- **Taille de la base vectorielle** : ~50-100 MB pour 8 documents
- **Chunks générés** : ~50-80 chunks pour 8 documents standards

## 🤝 Contribution

Pour améliorer le système :

1. Ajoutez de la documentation dans `data/`
2. Ajustez les paramètres dans `RAGConfig`
3. Améliorez les prompts dans `generator.py`
4. Optimisez les stratégies de chunking

## 📝 Licence

Projet développé pour le support technique TELNET SmartConnect.

## 🆘 Support

Pour toute question ou problème :
- Vérifiez la section Dépannage
- Consultez les logs de l'application
- Vérifiez la configuration Ollama

---

**Version** : 1.0  
**Date** : 2026-09-03  
**Statut** : Production Ready