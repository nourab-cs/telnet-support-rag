# Résumé du Projet TELNET Support Bot - RAG System

## 📋 Description

Système intelligent de support client TELNET SmartConnect basé sur RAG (Retrieval-Augmented Generation). Le système permet de rechercher dans la documentation technique et de générer des réponses contextuelles.

## 🏗️ Architecture

### Structure du Projet

```
telnet-support-bot/
├── src/                           # Code source modulaire
│   ├── __init__.py
│   ├── document_loader.py         # Chargement de documents (Markdown)
│   ├── text_cleaner.py           # Nettoyage et prétraitement
│   ├── chunker.py                # Découpage intelligent des documents
│   ├── embedder.py               # Génération d'embeddings (BGE-M3)
│   ├── chroma_store.py           # Base vectorielle ChromaDB
│   ├── retriever.py              # Récupération sémantique
│   ├── generator.py              # Génération de réponses (Mistral)
│   └── pipeline.py               # Pipeline RAG complet
├── data/                         # Documents source (8 fichiers Markdown)
│   ├── 01_guide_installation_smartconnect.md
│   ├── 02_faq_problemes_courants.md
│   ├── 03_documentation_api.md
│   ├── 04_guide_utilisateur_dashboard.md
│   ├── 05_procedure_maintenance.md
│   ├── 06_notes_version.md
│   ├── 07_configuration_reseau.md
│   └── 08_troubleshooting_guides.md
├── chroma_db/                    # Base vectorielle persistée
├── main.py                       # Point d'entrée principal (mode interactif)
├── test_rag.py                   # Script de test simple
├── evaluate_retrieval.py        # Évaluation de la retrieval
├── evaluate_rag.py               # Évaluation complète (avec LLM)
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation utilisateur
```

## 🔧 Technologies Utilisées

- **Python 3.14**
- **LangChain 1.3.15** - Framework LLM
- **LangChain Community 0.4.2** - Intégrations
- **LangChain Chroma 1.1.0** - Intégration ChromaDB
- **LangChain Ollama 1.1.0** - Intégration Ollama
- **ChromaDB 1.5.9** - Base vectorielle
- **Sentence Transformers** - Embeddings BGE-M3
- **Ollama** - LLM local (Mistral)

## 📊 État Actuel

### ✅ Fonctionnalités Implémentées

1. **Pipeline RAG complet**
   - Chargement de documents Markdown
   - Nettoyage et prétraitement du texte
   - Chunking intelligent (taille: 800, overlap: 150)
   - Génération d'embeddings (BGE-M3)
   - Stockage vectoriel persistant (ChromaDB)
   - Récupération sémantique (k=4 documents)
   - Génération de réponses (Mistral via Ollama)

2. **Interface utilisateur**
   - Mode conversationnel interactif
   - Affichage des sources utilisées
   - Script de test simple

3. **Évaluation**
   - Évaluation de la retrieval (sans LLM)
   - Mesure des temps de réponse
   - Analyse de la précision des sources
   - Couverture des mots-clés

### 📈 Résultats de l'Évaluation

**Performance Temporelle:**
- Temps moyen de retrieval: 0.272s
- Temps minimum: 0.198s
- Temps maximum: 0.516s

**Qualité de la Retrieval:**
- Précision de la source: 60.0% (3/5)
- Couverture moyenne des mots-clés: 76.0%
- Score global: 68.0% (Performance: BONNE)

### ⚠️ Limitations Actuelles

1. **LLM requis pour génération** - Ollama doit être démarré pour utiliser le mode complet
2. **Dépendance sur les noms de fichiers** - Différences singulier/pluriel affectent la précision
3. **Chunking statique** - Taille fixe sans adaptation dynamique
4. **Pas de mémoire conversationnelle** - Chaque question est indépendante

## 🚀 Utilisation

### Installation des Dépendances

```bash
pip install -r requirements.txt
```

### Lancer le Système

**Mode interactif:**
```bash
python main.py
```

**Test simple:**
```bash
python test_rag.py
```

**Évaluation retrieval:**
```bash
python evaluate_retrieval.py
```

**Évaluation complète (avec LLM):**
```bash
# D'abord démarrer Ollama dans un terminal séparé
ollama serve

# Puis lancer l'évaluation
python evaluate_rag.py
```

## 🔐 Configuration

Les paramètres sont configurés dans les fichiers correspondants:

- **Chunking**: `src/chunker.py` - chunk_size=800, chunk_overlap=150
- **Embeddings**: `src/embedder.py` - model="BAAI/bge-m3"
- **Retrieval**: `src/retriever.py` - k=4 documents
- **LLM**: `src/generator.py` - model="mistral", temperature=0.1
- **Base de données**: `src/chroma_store.py` - collection="telnet_support"

## 📝 Notes Importantes

1. **Avertissement de dépréciation** - `HuggingFaceEmbeddings` sera remplacé par `langchain-huggingface` dans le futur
2. **Base vectorielle persistée** - Le dossier `chroma_db/` contient les embeddings générés
3. **Documents Markdown uniquement** - Le document_loader est configuré pour les fichiers .md
4. **Encodage Windows** - Les caractères spéciaux ont été retirés pour compatibilité

## 🎯 Améliorations Possibles

1. **Architecture multi-agent** - 6 agents spécialisés comme prévu dans le projet original
2. **Mémoire conversationnelle** - Suivi du contexte sur plusieurs échanges
3. **Chunking dynamique** - Adaptation de la taille selon le type de document
4. **Reranking** - Re-ordonnancement des résultats pour meilleure précision
5. **Interface web** - Frontend React avec FastAPI
6. **Base de données tickets** - PostgreSQL pour historer les tickets
7. **Dashboard d'administration** - Statistiques et monitoring
8. **Tests automatisés** - Suite de tests unitaires et d'intégration

## 📅 Historique du Projet

- **Initialisation**: Création de la structure de base
- **Nettoyage**: Suppression des fichiers obsolètes et duplication
- **Modularisation**: Réorganisation en architecture modulaire `src/`
- **Correction des imports**: Adaptation aux nouvelles versions de LangChain
- **Évaluation**: Mise en place des scripts d'évaluation de performance
- **Documentation**: Création de la documentation utilisateur

## 🔗 Ressources

- **Documentation LangChain**: https://python.langchain.com/
- **ChromaDB**: https://docs.trychroma.com/
- **Ollama**: https://ollama.com/
- **BGE-M3**: https://huggingface.co/BAAI/bge-m3

---
*Projet nettoyé et fonctionnel - Version finale RAG*