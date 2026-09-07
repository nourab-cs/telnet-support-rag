# Guide utilisateur — Dashboard TELNET SmartConnect

## Connexion

1. Ouvrez votre navigateur à l'adresse : `https://<votre-serveur>:8443`
2. Saisissez votre email et mot de passe
3. Vous arrivez sur la page d'accueil du dashboard

## Page d'accueil

La page d'accueil affiche en un coup d'œil :
- **Nombre de devices en ligne / hors ligne**
- **Alertes actives** (en rouge si critiques)
- **Graphique temps réel** des 24 dernières heures
- **Carte géographique** des équipements

## Navigation

Le menu latéral donne accès à :

| Section | Description |
|---------|-------------|
| **Dashboard** | Vue d'ensemble temps réel |
| **Devices** | Liste de tous les équipements |
| **Données** | Historique et exports |
| **Alertes** | Configuration des notifications |
| **Administration** | Gestion utilisateurs et paramètres |

## Consulter les données d'un device

1. Cliquez sur **Devices** dans le menu
2. Utilisez les filtres en haut : type, statut, localisation
3. Cliquez sur un device pour voir ses détails
4. Onglet **Données** : graphique interactif, zoom, export
5. Onglet **Commandes** : envoyer une action (si device compatible)

## Créer une alerte

1. **Alertes > Nouvelle règle**
2. Choisissez le type :
   - **Seuil** : alerte si une valeur dépasse X
   - **Inactivité** : alerte si pas de données depuis X minutes
   - **Offline** : alerte si un device se déconnecte
3. Sélectionnez les devices concernés
4. Choisissez les destinataires (email, SMS, webhook)
5. **Sauvegarder**

## Filtres et recherche

Dans la liste des devices :
- Recherche par nom, ID, ou tag
- Filtrer par type (capteur, actionneur, passerelle)
- Filtrer par statut (online, offline, warning)
- Filtrer par localisation (région, site)

## Exports

Vous pouvez exporter vos données dans plusieurs formats :
- **CSV** : compatible Excel
- **JSON** : pour intégration avec d'autres outils
- **PDF** : pour reporting

Allez dans **Données > Exporter**, sélectionnez la période et le format.

## Personnalisation

Chaque utilisateur peut personnaliser :
- Langue (français, anglais, arabe)
- Fuseau horaire
- Thème (clair / sombre)
- Devices en favoris (épinglés en haut de la liste)

## Astuces

- **Raccourci clavier** : `Ctrl+K` ouvre la recherche rapide
- **Mode plein écran** : `F11` sur le dashboard pour masquer le menu
- **URL bookmarkable** : chaque filtre génère une URL que vous pouvez partager ou bookmarker
