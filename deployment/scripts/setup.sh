#!/bin/bash
set -e

# Log file setup
LOGFILE="/var/log/freefontfinder_setup.log"
mkdir -p "$(dirname $LOGFILE)"
exec 1> >(tee -a "$LOGFILE") 2>&1

echo "Starting Free Font Finder setup at $(date)"

# Function to log steps
log_step() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 1. System Setup
log_step "Setting up swap space..."
if [ ! -f /swapfile ]; then
    fallocate -l 1G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
fi

# 2. Install Dependencies
log_step "Installing system dependencies..."
apt-get update
apt-get install -y python3-pip python3-venv nginx certbot python3-certbot-nginx git

# 3. Application Setup
log_step "Setting up application directory..."
mkdir -p /var/www/freefontfinder/{static,model,logs}
cd /var/www/freefontfinder

# Clone repository if it doesn't exist
#if [ ! -d "/var/www/freefontfinder/.git" ]; then
#    git clone https://github.com/timholds/Font-Familiarity.git .
#else
#    git pull
#fi

# 4. Python Environment
log_step "Setting up Python environment..."
if [ -d "venv" ]; then
    rm -rf venv
fi
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r frontend_requirements.txt

# 5. Configuration Files
log_step "Setting up configuration files..."
#cp ../deployment/configs/nginx.conf /etc/nginx/sites-available/freefontfinder
cp /var/www/freefontfinder/deployment/configs/nginx.conf /etc/nginx/sites-available/freefontfinder
ln -sf /etc/nginx/sites-available/freefontfinder /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

cp /var/www/freefontfinder/deployment/configs/gunicorn.conf.py /var/www/freefontfinder/
cp /var/www/freefontfinder/deployment/configs/freefontfinder.service /etc/systemd/system/

# 6. Permissions
log_step "Setting permissions..."
mkdir -p /var/log/freefontfinder

chown -R www-data:www-data /var/www/freefontfinder
chown -R www-data:www-data /var/log/freefontfinder
chmod -R 755 /var/www/freefontfinder

# 7. Services
log_step "Starting services..."
systemctl daemon-reload
systemctl enable freefontfinder
systemctl restart freefontfinder
systemctl restart nginx

# 8. SSL Setup (only if domain is ready)
if host freefontfinder.com > /dev/null 2>&1; then
    log_step "Setting up SSL..."
    certbot --nginx -d freefontfinder.com -d www.freefontfinder.com --non-interactive --agree-tos --email your-email@example.com
else
    log_step "Skipping SSL setup - domain not ready"
fi

# 9. Memory Monitoring
log_step "Setting up memory monitoring..."
cat > /usr/local/bin/monitor_memory.sh << 'INNEREOF'
#!/bin/bash
MEMORY_USAGE=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
    echo "High memory usage alert: ${MEMORY_USAGE}%" | logger -t memory_monitor
fi
INNEREOF

chmod +x /usr/local/bin/monitor_memory.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/monitor_memory.sh") | crontab -

log_step "Setup completed successfully!"
echo "Memory Usage After Setup:"
free -h
echo "Swap Status:"
swapon --show