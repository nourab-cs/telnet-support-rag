# FAQ — Questions fréquentes sur TELNET SmartConnect

## Q1 : Comment réinitialiser le mot de passe administrateur ?

Connectez-vous en SSH sur le serveur et exécutez :

```bash
docker exec -it smartconnect python manage.py changepassword admin
```

Vous serez invité à saisir le nouveau mot de passe deux fois.

## Q2 : L'interface web ne charge pas, que faire ?

Vérifiez dans cet ordre :
1. Le service est bien démarré : `docker ps | grep smartconnect`
2. Le port est ouvert : `netstat -tlnp | grep 8443`
3. Pas d'erreur dans les logs : `docker logs smartconnect --tail 100`
4. Votre navigateur n'utilise pas un proxy qui bloque le HTTPS local

## Q3 : Comment ajouter un nouvel utilisateur ?

Via l'API :

```bash
curl -X POST https://localhost:8443/api/v1/users \
  -H "Authorization: Bearer VOTRE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@exemple.com", "role": "viewer"}'
```

Ou via l'interface : **Administration > Utilisateurs > Ajouter**.

## Q4 : Les données des capteurs ne remontent pas

Causes possibles :
- **Connectivité** : vérifiez que la passerelle a accès à Internet (`ping 8.8.8.8`)
- **MQTT** : le broker MQTT est-il démarré ? `docker ps | grep mqtt`
- **Credentials** : les identifiants MQTT sont corrects dans la configuration de la passerelle
- **Firewall** : le port 1883 (MQTT) est ouvert entre la passerelle et le serveur

## Q5 : Comment exporter mes données ?

Deux méthodes :
- **Interface** : **Données > Exporter > CSV/JSON**
- **API** : `GET /api/v1/devices/{device_id}/data?format=csv&from=2026-01-01`

## Q6 : Quelle est la différence entre les rôles "viewer", "operator" et "admin" ?

| Rôle | Lecture | Écriture | Administration |
|------|---------|----------|---------------|
| viewer | ✅ | ❌ | ❌ |
| operator | ✅ | ✅ | ❌ |
| admin | ✅ | ✅ | ✅ |

## Q7 : Comment mettre à jour SmartConnect vers une nouvelle version ?

```bash
# 1. Sauvegarder
docker exec smartconnect pg_dump -U sc_user smartconnect > backup.sql

# 2. Pull la nouvelle image
docker pull telnet/smartconnect:3.3.0

# 3. Arrêter l'ancien
docker stop smartconnect

# 4. Lancer le nouveau (les données sont dans le volume partagé)
docker run -d --name smartconnect-new \
  -p 8443:8443 \
  -v /opt/telnet/data:/var/lib/smartconnect \
  telnet/smartconnect:3.3.0
```

## Q8 : Les notifications email ne partent pas

Vérifiez :
1. Configuration SMTP dans **Paramètres > Notifications**
2. Le port 587 (TLS) n'est pas bloqué par votre pare-feu
3. Testez avec : `docker exec smartconnect python manage.py test_email --to=votre@email.com`
4. Vérifiez que vous n'êtes pas sur une liste noire (SPF/DKIM)

## Q9 : Puis-je installer SmartConnect sur Windows ?

Officiellement, seule la version Linux (Ubuntu/Debian) est supportée en production. Une version de développement existe pour Windows via WSL2, mais elle n'est pas recommandée pour un usage critique.

## Q10 : Comment contacter le support TELNET ?

- **Email** : support@telnet-holding.com
- **Téléphone** : +216 71 000 000 (8h-17h GMT+1)
- **Portail** : https://support.telnet-holding.com
- **Urgences 24/7** : uniquement pour les clients avec contrat SLA Premium
