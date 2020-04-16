import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import argparse
import configparser


parser = argparse.ArgumentParser(description='send emails from python.')
parser.add_argument('config', metavar='ini', type=str,
                   help='path to config file')
parser.add_argument('subject',  type=str,
                   help='subject')
parser.add_argument('message', type=str,
                   help='subject',default="")
parser.add_argument('map', type=str,nargs='?',
                   help='map',default="")

args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)
settings = config['Settings']

smtp_server = settings['smtp_server'].strip('"')
smtp_port = settings['smtp_port'].strip('"')
email_address = settings['email_address'].strip('"')
smtp_passwd = settings['smtp_passwd'].strip('"')
smtp_user = settings['email_address'].strip('"')
recipients = settings['email'].strip('"')

# set up the SMTP server
s = smtplib.SMTP(host=smtp_server, port=int(smtp_port))
s.starttls()
s.login(email_address, smtp_passwd)


msg = MIMEMultipart() 
msg['From']=email_address
msg['To']=recipients
msg['Subject']=args.subject



msg.attach(MIMEText(args.message, 'plain'))

if settings['send_map'].strip('"') == 'true':
    with open(args.map, 'rb') as fp:
        img = MIMEImage(fp.read())
        msg.attach(img)

try:
    s.send_message(msg)
    print('send')
except:
    print(sys.exc_info()[0])
    raise
finally:
    s.quit()
