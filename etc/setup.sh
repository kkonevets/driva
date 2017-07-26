#!/usr/bin/env bash
sudo chown -R raxel:raxel ~/driva


conda create --name driva
source activate driva

sudo rm /etc/nginx/conf.d/driva.conf

sudo ln -s /home/raxel/driva/driva.conf /etc/nginx/conf.d/
sudo /etc/init.d/nginx restart

sudo rm /home/raxel/driva/driva.sock
sudo uwsgi --ini /home/raxel/driva/driva.ini
sudo chown -R www-data:www-data ~/driva

sudo rm /etc/uwsgi/vassals/driva.ini
sudo ln -s ~/driva/driva.ini /etc/uwsgi/vassals

sudo restart uwsgi
sudo restart nginx

sudo vim /var/log/uwsgi/driva.log

ls -l .

sudo chown -R flask:flask /var/www/uploads
