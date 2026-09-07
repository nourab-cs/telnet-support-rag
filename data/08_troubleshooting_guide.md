# Guide de dépannage — TELNET SmartConnect

## Le service ne démarre pas

### Symptômes
- `docker ps` ne montre pas le conteneur `smartconnect`
- `docker logs smartconnect` affiche des erreurs

### Diagnostic
```bash
# Vérifier l'état détaillé
docker inspect smartconnect | jq '.[0].State'

# Voir les dernières erreurs
docker logs smartconnect --tail 100

# Vérifier les ports occupés
sudo lsof -i :8443
```

### Solutions courantes

**1. Port déjà utilisé**
```
Error: bind: address already in use
```
→ Arrêtez le service qui utilise le port : `sudo systemctl stop <service>`
→ Ou changez le port : `docker run -p 9000:8443 ...`

**2. Permissions insuffisantes**
```
Error: permission denied on /var/lib/smartconnect
```
→ Corrigez les permissions : `sudo chown -R 1000:1000 /opt/telnet/data`

**3. Clé de licence invalide**
```
Error: License key not valid for this version
```
→ Vérifiez le format : doit commencer par `TLNT-` et faire 32 caractères
→ Vérifiez que la licence n'a pas expiré dans **Admin > Licence**

## Les devices apparaissent hors ligne

### Diagnostic
1. **Vérifier la passerelle** : `ping <ip_passerelle>`
2. **Vérifier le broker MQTT** : `docker logs smartconnect | grep mqtt`
3. **Tester la connexion MQTT manuellement** :
   ```bash
   mosquitto_sub -h localhost -p 1883 -u device_001 -P <password> -t "telemetry/#" -v
   ```

### Solutions
- **Passerelle non connectée** : vérifiez l'alimentation et le réseau local
- **Credentials MQTT incorrects** : régénérez dans **Devices > device > Régénérer mot de passe**
- **Broker surchargé** : augmentez `max_connections` dans `mosquitto.conf`

## Lenteurs sur le dashboard

### Causes possibles
- **Trop de données en cache navigateur** : `Ctrl+Shift+R` pour hard refresh
- **Base de données non optimisée** : lancez `VACUUM ANALYZE`
- **Index manquants** : vérifiez avec `EXPLAIN ANALYZE` sur les requêtes lentes

### Requête utile pour identifier les requêtes lentes
```sql
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

## Erreurs API

### 401 Unauthorized
- Token expiré → renouveler avec `POST /auth/login`
- Token manquant → vérifier l'en-tête `Authorization`

### 429 Too Many Requests
- Vous dépassez 100 req/min
- Implémentez un **retry avec backoff exponentiel** :
  ```python
  import time
  for attempt in range(5):
      response = call_api()
      if response.status_code == 429:
          time.sleep(2 ** attempt)
      else:
          break
  ```

### 500 Internal Server Error
- Vérifiez les logs : `docker logs smartconnect --since 1h`
- Si c'est un bug, contactez le support avec :
  - L'endpoint appelé
  - Le body de la requête
  - L'heure exacte
  - L'ID de corrélation (présent dans le header `X-Request-ID`)

## Sauvegardes ne se font plus

```bash
# Vérifier le cron
crontab -l | grep backup

# Tester manuellement
/opt/telnet/scripts/backup.sh

# Vérifier l'espace de destination
df -h /backup/
```

## Performance dégradée après mise à jour

1. **Videz le cache Redis** : `docker exec smartconnect redis-cli FLUSHALL`
2. **Relancez les workers** : `docker restart smartconnect`
3. **Vérifiez la migration** : `docker exec smartconnect python manage.py showmigrations`

Si le problème persiste, restaurez la sauvegarde précédente et ouvrez un ticket.

## Besoin d'aide supplémentaire

Avant de contacter le support, préparez :

1. **Le bundle de diagnostic** :
   ```bash
   docker exec smartconnect python manage.py diag_bundle > diag_$(date +%Y%m%d).tar.gz
   ```

2. **Les informations** :
   - Version de SmartConnect (`/api/v1/health`)
   - OS et version
   - Nombre de devices
   - Description précise du problème
   - Étapes pour reproduire

3. **L'envoi** : support@telnet-holding.com avec objet `[BUG] description courte`

Temps de réponse moyen :
- **Critique (production down)** : 4h
- **Standard** : 24h ouvrées
- **Question générale** : 48h ouvrées
