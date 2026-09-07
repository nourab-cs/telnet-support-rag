# Configuration réseau — TELNET SmartConnect

## Ports réseau utilisés

| Port | Protocole | Usage | Direction |
|------|-----------|-------|-----------|
| 22 | SSH | Administration | Entrant |
| 443 | HTTPS | Interface web + API | Entrant |
| 1883 | MQTT | Communication devices | Entrant |
| 8883 | MQTTS | MQTT over TLS | Entrant |
| 5432 | PostgreSQL | Base de données | Local uniquement |
| 6379 | Redis | Cache | Local uniquement |
| 8443 | HTTPS | API secondaire (legacy) | Entrant |
| 9090 | Prometheus | Métriques | Local |
| 3000 | Grafana | Dashboards métriques | Entrant (optionnel) |

## Configuration firewall

### Avec UFW (Ubuntu)

```bash
# Politique par défaut
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH
sudo ufw allow 22/tcp

# Web et API
sudo ufw allow 443/tcp
sudo ufw allow 8443/tcp

# MQTT
sudo ufw allow 1883/tcp
sudo ufw allow 8883/tcp

# Grafana (optionnel, à restreindre à un sous-réseau)
sudo ufw allow from 192.168.1.0/24 to any port 3000

sudo ufw enable
sudo ufw status verbose
```

### Avec iptables

```bash
# Accepter les connexions established
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# SSH
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Web
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 8443 -j ACCEPT

# MQTT
iptables -A INPUT -p tcp --dport 1883 -j ACCEPT
iptables -A INPUT -p tcp --dport 8883 -j ACCEPT

# Loopback
iptables -A INPUT -i lo -j ACCEPT

# Bloquer le reste
iptables -A INPUT -j DROP
```

## Configuration Nginx (reverse proxy)

```nginx
server {
    listen 443 ssl http2;
    server_name smartconnect.votredomaine.com;

    ssl_certificate /etc/letsencrypt/live/.../fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/.../privkey.pem;

    # SSL moderne
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Sécurité headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    # Taille max upload
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8443;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://127.0.0.1:8443;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Configuration MQTT broker

Le broker Mosquitto est intégré à SmartConnect. Sa configuration se trouve dans `/opt/telnet/data/config/mosquitto.conf` :

```conf
listener 1883
allow_anonymous false
password_file /opt/telnet/data/config/mosquitto_passwd

listener 8883
cafile /etc/ssl/certs/ca-certificates.crt
certfile /etc/letsencrypt/live/.../fullchain.pem
keyfile /etc/letsencrypt/live/.../privkey.pem
```

## VPN recommandé pour les sites distants

Pour les déploiements sur des sites distants (sans IP publique), utilisez **WireGuard** :

```bash
# Installation
sudo apt install wireguard

# Génération des clés
wg genkey | tee privatekey | wg pubkey > publickey

# Configuration serveur
sudo nano /etc/wireguard/wg0.conf
```

```conf
[Interface]
Address = 10.0.0.1/24
PrivateKey = <SERVER_PRIVATE_KEY>
ListenPort = 51820

[Peer]
PublicKey = <CLIENT_PUBLIC_KEY>
AllowedIPs = 10.0.0.2/32
```

## DNS

Configurez un nom de domaine pointant vers votre serveur :
- `smartconnect.votredomaine.com` → dashboard et API
- `mqtt.votredomaine.com` → broker MQTT

Utilisez **Let's Encrypt** pour les certificats :

```bash
sudo apt install certbot
sudo certbot certonly --nginx -d smartconnect.votredomaine.com
```

## Surveillance réseau

Pour monitorer le trafic :

```bash
# Installer iftop
sudo apt install iftop
sudo iftop -i eth0

# Vérifier les connexions actives
ss -tunap | grep smartconnect
```

## Problèmes réseau fréquents

- **MQTT ne reçoit rien** : port 1883 bloqué par le FAI ? Utilisez 8883 (TLS) ou un port non standard
- **API lente** : vérifiez que PostgreSQL n'est pas sur un autre VLAN (latence réseau)
- **Certificats expirés** : `sudo certbot renew` (renouvellement auto via cron)
