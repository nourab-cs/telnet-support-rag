# Documentation API — TELNET SmartConnect REST API v1

**Base URL** : `https://<serveur>:8443/api/v1`
**Authentification** : Bearer Token (JWT)
**Format** : JSON

## Authentification

Toutes les requêtes nécessitent un token JWT dans l'en-tête :

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Pour obtenir un token :

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@exemple.com",
  "password": "VOTRE_MDP"
}
```

Réponse :

```json
{
  "token": "eyJhbGciOi...",
  "expires_at": "2026-08-15T12:00:00Z",
  "user": { "id": "u_123", "role": "operator" }
}
```

## Endpoints principaux

### Devices

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/devices` | Liste tous les devices |
| GET | `/devices/{id}` | Détails d'un device |
| POST | `/devices` | Créer un device |
| DELETE | `/devices/{id}` | Supprimer un device |
| GET | `/devices/{id}/data` | Données d'un device |
| POST | `/devices/{id}/command` | Envoyer une commande |

### Exemple : lister les devices

```bash
curl -X GET https://api.telnet.com/api/v1/devices \
  -H "Authorization: Bearer VOTRE_TOKEN"
```

Réponse :

```json
{
  "devices": [
    {
      "id": "dev_001",
      "name": "Capteur température hangar A",
      "type": "temperature_sensor",
      "status": "online",
      "last_seen": "2026-07-29T13:45:00Z"
    }
  ],
  "total": 47,
  "page": 1
}
```

### Exemple : récupérer les données d'un capteur

```bash
curl -X GET "https://api.telnet.com/api/v1/devices/dev_001/data?from=2026-07-28&to=2026-07-29" \
  -H "Authorization: Bearer VOTRE_TOKEN"
```

### Exemple : envoyer une commande à un actionneur

```bash
curl -X POST https://api.telnet.com/api/v1/devices/act_005/command \
  -H "Authorization: Bearer VOTRE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action": "turn_on", "duration": 300}'
```

## Codes d'erreur

| Code | Signification |
|------|---------------|
| 200 | Succès |
| 201 | Ressource créée |
| 400 | Requête invalide (paramètres manquants ou mal formés) |
| 401 | Token manquant ou expiré |
| 403 | Permissions insuffisantes |
| 404 | Ressource non trouvée |
| 429 | Trop de requêtes (rate limit : 100 req/min) |
| 500 | Erreur serveur |

## Rate limiting

L'API est limitée à **100 requêtes par minute par token**. Au-delà, vous obtenez un code 429. Les headers de réponse indiquent votre consommation :

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1690540800
```

## Webhooks

Vous pouvez recevoir des notifications en temps réel :

```http
POST /api/v1/webhooks
{
  "url": "https://votresite.com/webhook",
  "events": ["device.offline", "alert.triggered"],
  "secret": "VOTRE_SECRET"
```

## SDK officiels

- **Python** : `pip install telnet-smartconnect`
- **Node.js** : `npm install @telnet/smartconnect`
- **Java** : disponible sur Maven Central (`com.telnet:smartconnect-sdk`)

## Limites de l'API

- Maximum 10 000 devices par tenant
- Données historiques conservées 2 ans
- WebSocket temps réel : max 100 connexions simultanées par token
