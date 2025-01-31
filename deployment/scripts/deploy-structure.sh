#!/bin/bash
# Run this locally to set up deployment directory structure
# Update this section in deployment/scripts/deploy-structure.sh
#log_step "Creating initial directory structure..."

# Create deployment directory structure
mkdir -p deployment/{scripts,configs}

# Create main setup script
cat > deployment/scripts/setup.sh << 'EOL'
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
if [ ! -d "/var/www/freefontfinder/.git" ]; then
    git clone https://github.com/timholds/Font-Familiarity.git .
else
    git pull
fi

# 4. Python Environment
log_step "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt

# 5. Configuration Files
log_step "Setting up configuration files..."
cp ../deployment/configs/nginx.conf /etc/nginx/sites-available/freefontfinder
ln -sf /etc/nginx/sites-available/freefontfinder /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

cp ../deployment/configs/gunicorn.conf.py ./
cp ../deployment/configs/freefontfinder.service /etc/systemd/system/

# 6. Permissions
log_step "Setting permissions..."
chown -R www-data:www-data /var/www/freefontfinder
chown -R www-data:www-data /var/log/freefontfinder
chmod -R 755 /var/www/freefontfinder

# 7. Services
log_step "Starting services..."
systemctl daemon-reload
systemctl enable freefontfinder
systemctl restart freefontfinder
systemctl restart nginx

# 8. SSL Setup
log_step "Setting up SSL with certbot..."
apt-get install -y certbot python3-certbot-nginx

# Only proceed with SSL setup if DNS is properly configured
if host freefontfinder.com > /dev/null 2>&1; then
    log_step "DNS check passed, proceeding with SSL setup..."
    certbot --nginx \
        -d freefontfinder.com \
        -d www.freefontfinder.com \
        --non-interactive \
        --agree-tos \
        --redirect \
        --email your-email@example.com \
        --post-hook "systemctl restart nginx"
        
    # Verify SSL setup
    if curl -s -I https://freefontfinder.com | grep -q "200 OK"; then
        log_step "SSL setup successful!"
    else
        log_step "Warning: SSL setup might not be complete. Please check configuration."
    fi
else
    log_step "DNS not yet configured for freefontfinder.com - skipping SSL setup"
    log_step "Run: certbot --nginx -d freefontfinder.com -d www.freefontfinder.com once DNS is ready"
fi
