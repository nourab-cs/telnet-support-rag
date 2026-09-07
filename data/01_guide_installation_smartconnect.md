# Guide d'installation — TELNET SmartConnect v3.2

**Produit** : Plateforme IoT SmartConnect
**Version** : 3.2.1
**Date** : 2026-03-15
**Public** : administrateurs système, intégrateurs

## 1. Prérequis système

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| CPU | 4 cœurs | 8 cœurs |
| RAM | 8 Go | 16 Go |
| Disque | 50 Go SSD | 100 Go SSD |
| Python | 3.10 | 3.11 |
| Docker | 24.0+ | 26.0+ |

## 2. Installation via Docker (recommandée)

```bash
# 1. Récupérer l'image officielle
docker pull telnet/smartconnect:3.2.1

# 2. Créer le réseau interne
docker network create telnet-net

# 3. Lancer le conteneur
docker run -d \
  --name smartconnect \
  --network telnet-net \
  -p 8443:8443 \
  -v /opt/telnet/data:/var/lib/smartconnect \
  -e SC_LICENSE_KEY="VOTRE_CLE" \
  telnet/smartconnect:3.2.1
```

## 3. Installation manuelle (avancée)

Si vous ne pouvez pas utiliser Docker :

```bash
# Installation des dépendances
sudo apt update
sudo apt install -y python3.11 python3-pip postgresql-14 nginx

# Clonage du dépôt
git clone https://github.com/telnet/smartconnect.git
cd smartconnect
pip install -r requirements.txt

# Configuration de la base
sudo -u postgres createdb smartconnect
python manage.py migrate
python manage.py createsuperuser
```

## 4. Vérification de l'installation

Une fois installé, vérifiez que tout fonctionne :

```bash
curl -k https://localhost:8443/api/v1/health
# Réponse attendue : {"status": "ok", "version": "3.2.1"}
```

Si vous obtenez un timeout, vérifiez que le port 8443 est bien ouvert :

```bash
sudo ufw allow 8443/tcp
sudo ufw status
```

## 5. Configuration du premier tenant

1. Connectez-vous à l'interface d'administration : `https://<votre-ip>:8443/admin`
2. Créez un nouveau tenant via le menu **Tenants > Nouveau**
3. Notez l'**API Key** générée — elle ne sera plus jamais affichée
4. Configurez les paramètres SMTP pour les notifications

## Problèmes fréquents à l'installation

- **Erreur "port already in use"** : un autre service utilise le 8443. Changez le port avec `-p 9000:8443`
- **Erreur de licence** : vérifiez que votre clé commence bien par `TLNT-`
- **Performance lente** : augmentez la RAM allouée au conteneur Docker avec `--memory=8g`
