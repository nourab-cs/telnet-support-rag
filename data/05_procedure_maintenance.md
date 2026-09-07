# Procédure de maintenance — TELNET SmartConnect

**Fréquence recommandée** : mensuelle
**Durée estimée** : 1 à 2 heures
**Prérequis** : accès SSH au serveur, droits sudo

## 1. Sauvegarde avant maintenance

```bash
# Sauvegarde de la base de données
docker exec smartconnect pg_dump -U sc_user smartconnect | gzip > backup_$(date +%Y%m%d).sql.gz

# Sauvegarde des fichiers de configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz /opt/telnet/data/config/

# Sauvegarde des volumes Docker
docker run --rm --volumes-from smartconnect -v $(pwd):/backup ubuntu tar cvf /backup/smartconnect_volumes.tar /var/lib/smartconnect
```

Conservez ces sauvegardes dans un emplacement sécurisé (S3, NAS externe).

## 2. Vérification de l'état du système

```bash
# Espace disque
df -h /opt/telnet/

# Mémoire
free -h

# Charge CPU
uptime

# Logs récents
docker logs smartconnect --since 24h | tail -50
```

Si l'espace disque est inférieur à 10%, planifiez un nettoyage :

```bash
# Nettoyer les vieux logs
docker exec smartconnect find /var/log -name "*.log.*" -mtime +30 -delete

# Nettoyer les données temporaires
docker exec smartconnect rm -rf /tmp/*
```

## 3. Mise à jour de l'image (si applicable)

```bash
# Pull de la dernière version stable
docker pull telnet/smartconnect:stable

# Vérification du changelog
curl -s https://api.telnet.com/releases/stable | jq .
```

## 4. Nettoyage de la base vectorielle

Si vous utilisez ChromaDB :

```bash
docker exec smartconnect python manage.py clean_embeddings --older-than 365
```

## 5. Vérification de la sécurité

```bash
# Vérifier que les certificats ne sont pas expirés
openssl s_client -connect localhost:8443 < /dev/null 2>/dev/null | openssl x509 -noout -dates

# Vérifier les utilisateurs inactifs (à désactiver)
docker exec smartconnect python manage.py list_inactive_users --days 90
```

## 6. Tests post-maintenance

```bash
# Health check
curl -k https://localhost:8443/api/v1/health

# Test de bout en bout
docker exec smartconnect python manage.py smoke_test
```

## 7. Rotation des logs

Configurez logrotate dans `/etc/logrotate.d/smartconnect` :

```
/opt/telnet/data/logs/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    postrotate
        docker exec smartconnect kill -HUP 1
    endscript
}
```

## 8. Surveillance post-maintenance

Pendant 24h après la maintenance, surveillez :
- **CPU/RAM** via `htop` ou le dashboard
- **Erreurs 5xx** dans les logs
- **Latence API** via les métriques Prometheus

## En cas de problème

Si quelque chose se casse pendant la maintenance :
1. **Ne paniquez pas** : les sauvegardes sont là
2. **Restaurer la base** : `gunzip < backup_YYYYMMDD.sql.gz | docker exec -i smartconnect psql -U sc_user smartconnect`
3. **Contacter le support** : support@telnet-holding.com avec le code `MAINT-FAIL`
