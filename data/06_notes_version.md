# Notes de version — TELNET SmartConnect

## Version 3.3.0 (juin 2026)

### Nouveautés
- ✨ Support natif de **LoRaWAN** pour les passerelles longue portée
- ✨ Nouveau **dashboard temps réel** avec WebSocket (latence < 100ms)
- ✨ Module **d'alertes avancées** : conditions composées (ET/OU)
- ✨ **API v2 beta** : support de la pagination cursor-based

### Améliorations
- ⚡ Performances : 40% plus rapide sur les requêtes avec 10k+ devices
- ⚡ Réduction de 30% de la consommation mémoire
- 🎨 Refonte de l'interface d'administration

### Corrections de bugs
- 🐛 Fix : crash aléatoire du broker MQTT sous forte charge (#1247)
- 🐛 Fix : timezone incorrect dans les exports CSV pour fuseaux exotiques (#1251)
- 🐛 Fix : l'export PDF ne gérait pas les graphes avec > 1000 points

### Breaking changes
- ⚠️ L'endpoint `GET /api/v1/devices` retourne maintenant un format paginé. Les clients doivent gérer `page` et `per_page`.

## Version 3.2.1 (mars 2026) — actuelle

### Nouveautés
- ✨ Support de **PostgreSQL 15** et **16**
- ✨ Authentification **SSO via SAML 2.0** pour les entreprises
- ✨ Module de **rapports planifiés** (quotidien, hebdomadaire, mensuel)

### Corrections
- 🐛 Fix : fuite mémoire dans le worker d'embeddings (#1198)
- 🐛 Fix : les webhooks ne retry pas après une erreur 502

## Version 3.2.0 (janvier 2026)

### Nouveautés
- ✨ **Multi-tenancy** : isolation complète des données par client
- ✨ Module de **géolocalisation** avec support des cartes offline
- ✨ **RBAC** : permissions granulaires par ressource

### Breaking changes
- ⚠️ Le format de stockage des credentials a changé. Une migration automatique est lancée à la première installation.

## Version 3.1.0 (novembre 2025)

- ✨ Support du protocole **MQTT 5**
- ✨ Nouvelle API de **commandes temps réel**
- ✨ **Sauvegardes automatiques** chiffrées vers S3

## Version 3.0.0 (septembre 2025) — refonte majeure

- 🚀 Refonte complète de l'architecture (passage en microservices)
- 🚀 Nouveau moteur d'**anomalies** basé sur l'IA
- 🚀 Support de **Kubernetes** en natif
- ⚠️ Migration obligatoire depuis 2.x (script fourni)

## Roadmap 2026 (prévisionnel)

- **Q3 2026** : Assistant conversationnel IA intégré (c'est ton stage !)
- **Q4 2026** : Module de **maintenance prédictive** (ML sur les données capteurs)
- **Q1 2027** : Marketplace de plugins tiers
