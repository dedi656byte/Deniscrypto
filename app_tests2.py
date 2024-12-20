#import eventlet
#eventlet.monkey_patch()

from coinbase.rest import RESTClient
from dotenv import load_dotenv
from threading import Thread
import threading
import logging
import time
import uuid
from decimal import Decimal, getcontext, ROUND_DOWN
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from functools import wraps
import pyotp
from flask_mail import Mail, Message
import requests
from datetime import datetime
from flask_socketio import SocketIO
##############################################
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Input
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import base64

#######################################################################################################################
# Set decimal precision
getcontext().prec = 10

# Load environment variables from .env file
load_dotenv()

# Hardcoded password for login
HARDCODED_PASSWORD = os.getenv('LOGIN_PASSWORD')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load API credentials
api_key = os.getenv('COINBASE_API_KEY_ID')

# Load the private key from the PEM file
private_key_path = 'coinbase_private_key.pem'
with open(private_key_path, 'r') as key_file:
    api_secret = key_file.read()
# Create the RESTClient instance
client = RESTClient(api_key=api_key, api_secret=api_secret)

try:
    # Simple call to test authentication
    accounts = client.get_accounts()
    for account in accounts['accounts']:
        print("Successfully authenticated. Accounts data:", account['name'])

except Exception as e:
    print("Authentication failed:", e)
#####################################################################################################
#selected_crypto_pairs = ['BTC-USDC', 'ETH-USDC', 'USDT-USDC', 'ADA-USDC', 'DOGE-USDC', 'LTC-USDC', 'XRP-USDC', 'SOL-USDC']
#selected_crypto_pairs = ['BTC-USDC']
selected_crypto_pairs = []
#valide
#selected_crypto_pairs=['ADA-USDC','AAVE-USDC','ALGO-USDC','ARB-USDC','AVAX-USDC','BCH-USDC','BTC-USDC','CRV-USDC','DOGE-USDC','DOT-USDC','ETC-USDC','ETH-USDC','FET-USDC','FIL-USDC','GRT-USDC','HBAR-USDC','ICP-USDC','IDEX-USDC','LINK-USDC','LTC-USDC','MATIC-USDC','NEAR-USDC','PEPE-USDC','SOL-USDC','SUI-USDC','SUPER-USDC','SUSHI-USDC','SWFTC-USDC','UNI-USDC','USDT-USDC','VET-USDC','XLM-USDC','XYO-USDC','XRP-USDC','YFI-USDC']

#VALIDE
# Fetch product details
for selected_crypto_pair in selected_crypto_pairs:
    product_info = client.get_product(selected_crypto_pair)  # Utilisation correcte de 'pair' au lieu de 'selected_crypto_pair'

    # Extraction de la taille minimale de l'échange
    base_min_size = float(product_info['base_min_size'])

    #Décommentez cette ligne si vous avez besoin de l'incrément de la cotation
    quote_increment = float(product_info['quote_increment'])

    print(f"Base Minimum Size for {selected_crypto_pair}: {base_min_size}")
    # Décommentez cette ligne si vous avez besoin d'afficher l'incrément de la cotation
    print(f"Quote Increment for {selected_crypto_pair}: {quote_increment}")

####################################################################################################################################################
# Initialisation de Flask-SocketIO
app = Flask(__name__)
#socketio = SocketIO(app, async_mode='eventlet')

Profit_cumul = 0
log_data = ""  # Global log data
log_data1 = ""  # Global log data
log_data2 = ""
log_data3 = ""
log_data4 = ""
# Initialisation du client Coinbase
accounts = client.get_accounts()
####################################################################################################################################################
# Configuration de Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.elasticemail.com'
app.config['MAIL_PORT'] = 2525
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_DEBUG'] = True
app.config['MAIL_USERNAME'] = os.getenv("SENDER_EMAIL")  # Utilisez l'email de l'expéditeur
app.config['MAIL_PASSWORD'] = os.getenv("SENDER_PASSWORD")  # Mot de passe de l'email ou mot de passe spécifique à l'application
app.config['MAIL_DEFAULT_SENDER'] = os.getenv("SENDER_EMAIL")
mail = Mail(app)
# Configurer le générateur de code 2FA
totp = pyotp.TOTP(os.getenv("SECRET_KEY2"))  # Clé secrète pour générer les codes 2FA (à stocker de manière sécurisée)
current_2fa_code = None  # Variable pour stocker le code 2FA généré
user_email = os.getenv("USER_EMAIL")  # L'email du destinataire du code 2FA (peut être dynamique)
####################################################################################################################################################
app.secret_key = 'your_secret_key'  # Set a secret key for sessions

# Configurations
buy_percentage_of_capital = Decimal("0.05")  # 5% of capital per DCA buy
#sell_percentage_of_capital = Decimal("0.05") # 5% of capital per DCA sell
sell_profit_target = Decimal("0.005")  # Sell when 5% profit target is reached
stop_loss_threshold = Decimal("0.005")  # Stop loss at 5% below initial buy-in
dca_interval_minute = 1
dca_interval_seconds = dca_interval_minute * 60  # DCA interval in seconds (adjust as needed)
ia = False
####################################################################################################################################################
ADA_USDC= True
AAVE_USDC= True
AERO_USDC= True #supporte pas tradin avec IA
ALGO_USDC= True
AMP_USDC= True #supporte pas tradin avec IA
ARB_USDC= True
AVAX_USDC= True
BCH_USDC= True
BONK_USDC= True #supporte pas tradin avec IA
BTC_USDC= True
CRV_USDC= True
DOGE_USDC= True
DOT_USDC= True
ETH_USDC= True
EURC_USDC= True #supporte pas tradin avec IA
FET_USDC= True
FIL_USDC= True
GRT_USDC= True
HBAR_USDC= True
ICP_USDC= True
IDEX_USDC= True
INJ_USDC= True #supporte pas tradin avec IA
JASMY_USDC= True #supporte pas tradin avec IA
JTO_USDC= True #supporte pas tradin avec IA
LINK_USDC= True
LTC_USDC= True
MOG_USDC= True #supporte pas tradin avec IA
NEAR_USDC= True
ONDO_USDC= True #supporte pas tradin avec IA
PEPE_USDC= True
RENDER_USDC= True #supporte pas tradin avec IA
RNDR_USDC= True #supporte pas tradin avec IA
SEI_USDC= True #supporte pas tradin avec IA
SHIB_USDC= True #supporte pas tradin avec IA
SOL_USDC= True
SUI_USDC= True
SUPER_USDC= True
SUSHI_USDC= True
SWFTC_USDC= True
TIA_USDC= True #supporte pas tradin avec IA
UNI_USDC= True
USDT_USDC= True
VET_USDC= True
WIF_USDC= True #supporte pas tradin avec IA
XLM_USDC= True
XYO_USDC= True
XRP_USDC= True
YFI_USDC= True

ETC_USDC= True
MATIC_USDC= True


####################################################################################################################################################
bot_running = False
logs = []
#####################################################################################################
#VALIDE
from datetime import datetime
def log_message(message):
    global logs
    timestamped_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
    logs.append(timestamped_message)
    logging.info(timestamped_message)

def save_logs_to_file():
    file_path = os.path.join(os.getcwd(), 'logs.txt')
    with open(file_path, 'w') as file:
        for log in logs:
            file.write(log + '\n')
#####################################################################################################
#################################################################################################################################################################################
#LEPORSSAI
def fetch_crypto_data(crypto_pair, limit=500):
    fsym, tsym = crypto_pair.split('-')
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(f"{endpoint}?fsym={fsym}&tsym={tsym}&limit={limit}")
    data = pd.DataFrame(json.loads(res.content)['Data'])
    data = data.set_index('time')
    data.index = pd.to_datetime(data.index, unit='s')
    data = data.drop(['conversionType', 'conversionSymbol'], axis=1)
    return data

def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('prix', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    return (df - df.min()) / (df.max() - df.min())

def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'

all_predictions = {}
for crypto_pair in selected_crypto_pairs:
    print(f"Processing {crypto_pair}")

    hist = fetch_crypto_data(crypto_pair)

    train, test, X_train, X_test, y_train, y_test = prepare_data(hist, 'close', window_len=window_len, zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    targets = test['close'][window_len:]
    preds = model.predict(X_test).squeeze()

    mae = mean_absolute_error(preds, y_test)
    print(f"Mean Absolute Error for {crypto_pair}: {mae}")

    preds = test['close'].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    all_predictions[crypto_pair] = preds

    line_plot(targets, preds, 'actual', 'prediction', lw=3, title=f'{crypto_pair} Price Prediction')


def compare_first_real_and_last_pred(yesterday_last_real, today_pred):
    yesterday_last_value = yesterday_last_real.iloc[0]
    last_pred_value = today_pred.iloc[-1]
    if last_pred_value > yesterday_last_value:
        return f"Le prix de la crypto va augmenter (Today_Prediction: {last_pred_value} > Yesterday_Prediction: {yesterday_last_value})"
    else:
        return f"Le prix de la crypto va diminuer (Today_Prediction: {last_pred_value} < Yesterday_Prediction: {yesterday_last_value})"


# Fonction pour déterminer si une paire va monter ou descendre
def will_crypto_increase_or_decrease(yesterday_last_real, today_pred):
        yesterday_last_value = yesterday_last_real.iloc[0]
        last_pred_value = today_pred.iloc[-1]
        if last_pred_value > yesterday_last_value:
            return 1
            #return f"La paire {crypto_pair} va probablement augmenter (Prédiction: {last_predicted_price} > Réel: {last_real_price})"
        else:
            return 0
            #return f"La paire {crypto_pair} va probablement diminuer (Prédiction: {last_predicted_price} < Réel: {last_real_price})"


today_date = pd.to_datetime('today').normalize()
# Date précédente
previous_date = today_date - pd.Timedelta(days=1)

for crypto_pair in selected_crypto_pairs:
    hist = fetch_crypto_data(crypto_pair)
    train, test = train_test_split(hist, test_size=test_size)
    train, test, X_train, X_test, y_train, y_test = prepare_data(hist, 'close', window_len=window_len, zero_base=zero_base, test_size=test_size)
    targets = test['close'][window_len:]
    preds = model.predict(X_test).squeeze()
    preds = test['close'].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)


    #yesterday_last_real = test['close'].loc[test.index.date == previous_date.date()]
    yesterday_last_real = preds.loc[previous_date: previous_date]
    today_pred = preds.loc[today_date: today_date]

    # Comparaison des valeurs
    trend_comparison = compare_first_real_and_last_pred(yesterday_last_real, today_pred)
    #log_message(f"{crypto_pair} trend: {trend_comparison}")
    save_logs_to_file()
    print(f"{crypto_pair} trend: {trend_comparison}")
#LEPORSSAI
#################################################################################################################################################################################

#####################################################################################################
#VERIFIER SI CES FONCTION FONCTIONNE AVEC LA BONNE LOGIQUE
#####################################################################################################
#####################################################################################################
#VALIDE
def get_account_balance(selected_crypto_pair):
    global accounts
    """Fetch the account balance in the selected cryptocurrency."""
    try:
        selected_crypto = selected_crypto_pair.split('-')[0]
        log_message(f"Récupération du solde {selected_crypto}...")
        save_logs_to_file()
        #accounts = client.get_accounts()
        for account in accounts['accounts']:
            if account['currency'] == selected_crypto:
                balance = Decimal(account['available_balance']['value'])
                log_message(f"Solde trouvé: {balance} {selected_crypto}")
                save_logs_to_file()
                return balance
    except Exception as e:
        log_message(f"Erreur lors de la récupération du solde {selected_crypto}: {e}")
        save_logs_to_file()
    return Decimal('0')
#####################################################################################################
#VALIDE
def get_usdc_balance():
    global accounts
    """Fetch the USDC balance."""
    log_message(f"Nous procedons à la récupération du solde USDC....")
    save_logs_to_file()
    try:
        #accounts = client.get_accounts()
        for account in accounts['accounts']:
            if account['currency'] == 'USDC':
                return Decimal(account['available_balance']['value'])
    except Exception as e:
        log_message(f"Erreur Lors de la récupération du solde USDC: {e}")
        save_logs_to_file()
    return Decimal('0')
#####################################################################################################
#R.A.S
def get_market_price(product_id):
    """Fetch the latest market price for a given product."""
    try:
        market_data = client.get_market_trades(product_id=product_id, limit=1)
        #log_message(f"Nous recherchons le prix du {product_id} sur le marché .")
        #if 'trades' in market_data and market_data['trades']:
        price = Decimal(market_data['trades'][0]['price'])
        log_message(f"le prix actuel du {product_id} sur le marché est: {price} USDC")
        save_logs_to_file()
        return price
    except Exception as e:
        log_message(f"Error fetching market price for {product_id}: {e}")
        save_logs_to_file()
    return None
#####################################################################################################
#VALIDE
def check_usdc_balance():
    global accounts
    try:
        log_message("Vérification du solde USDC")
        save_logs_to_file()
        #accounts = client.get_accounts()
        for account in accounts['accounts']:
            if account['currency'] == 'USDC':
                solde_usdc = Decimal(account['available_balance']['value'])
                log_message(f"Solde USDC: {solde_usdc}")
                save_logs_to_file()
                return solde_usdc
        log_message("Aucun solde USDC trouvé")
        save_logs_to_file()
        return Decimal('0')
    except Exception as e:
        log_message(f"Erreur lors de la vérification du solde USDC: {e}")
        save_logs_to_file()
        return Decimal('0')
#####################################################################################################
#MODIFIER CETTE FONCTION POUR QUELLE RESSEMBLE A L'ANCIENNE VERSION DU PROJET
def place_market_buy(product_id):
    """Place a market buy order using USDC balance."""
    global buy_percentage_of_capital
    log_message(f"Vérifions le solde USDC")
    save_logs_to_file()
    usdc_balance = check_usdc_balance()  # Fetch USDC balance directly

    #verifions s il y a assez de usdc
    if usdc_balance <= Decimal('0.01'):
        log_message("No USDC balance available to place the order. message from place_market_buy function.")
        save_logs_to_file()
        return None

    try:
        # Calculate the amount in USDC based on available balance
        log_message(f"Calculons la quantité de USDC en se basant sur le solde disponible ")
        save_logs_to_file()
        effective_usdc_amount = usdc_balance * Decimal(buy_percentage_of_capital)

        # Fetch product details (including min trade size and precision)
        product_info = client.get_product(product_id)
        #base_min_size = Decimal(product_info['quote_min_size'])
        base_increment = product_info['quote_increment']

        # Ensure effective_btc_amount is above minimum trade size
        #if effective_usdc_amount < base_min_size:
        #    effective_usdc_amount = base_min_size

        # Apply rounding to match the precision level expected by Coinbase
        precision = int(base_increment.find('1') - 1)
        effective_usdc_amount1 = effective_usdc_amount.quantize(Decimal('1.' + '0' * precision), rounding=ROUND_DOWN)

        # Ensure effective_btc_amount is above the minimum trade size after rounding
        #if effective_usdc_amount1 < base_min_size:
            #effective_usdc_amount1 = base_min_size

        # Ensure effective amount is greater than zero after rounding
        if effective_usdc_amount1 <= Decimal('0'):
            log_message(f"Le montant ajusté de USDC après arrondi est trop faible: {effective_usdc_amount1}. annuler l'ordre.")
            save_logs_to_file()
            return None

        # Convert the effective amount to a fixed-point string without scientific notation
        formatted_usdc_amount = f"{effective_usdc_amount1:.{precision}f}"

        # Log the adjusted amount in BTC
        log_message(f"Montant ajusté de USDC avec précision : {formatted_usdc_amount}")
        save_logs_to_file()

        # Define the required individual arguments for create_order
        client_order_id = str(uuid.uuid4())  # Generate a unique client order ID
        side = "BUY"
        order_configuration = {
            "market_market_ioc": {
                "quote_size": formatted_usdc_amount  # Ensure this is a string for the API call
            }
        }
        #recuperons le prix au moment de l'achat de la paire en usdc
        prix_moment_achat = get_market_price(product_id)
        # Place the order with required arguments
        response = client.create_order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=side,
            order_configuration=order_configuration
        )
        #base_currency = product_id.split("-")[1]
        if response['success']:
            log_message("Order created successfully.")
            save_logs_to_file()
            log_message(f"prix unitaire à l'achat :{prix_moment_achat}: USDC")
            save_logs_to_file()
            log_message(f"Market buy order response for {product_id}: {response}")
            save_logs_to_file()
            # After buying, start monitoring for take profit or stop loss
            log_message("Lancons le suivie du stop loss et du take profit.")
            save_logs_to_file()
            log_message("=====================================================================")
            save_logs_to_file()
            monitor_position_for_tp_sl(product_id, effective_usdc_amount1, prix_moment_achat)
        else:
            #error_code = response['response']['error']
            #error_message = response['response']['message']
            #log_message(f"Failed to create order: {error_code} - {error_message}")
            #if error_code == 'INSUFFICIENT_FUND':
               #log_message(f"Error: Solde insuffisant. SVP veuillez approvisionner votre portefeuille {base_currency}.")
            #else:
                #log_message(f"Failed to create order: {error_code} - {error_message}")
            log_message(f"{response}.")
            save_logs_to_file()
        return response


    except Exception as e:
        log_message(f"Error placing market buy order for {product_id}: {e}")
        save_logs_to_file()
    return None
#####################################################################################################
#S IL YA PROBLEME VOIRE CETTE PARTIE
def monitor_position_for_tp_sl(product_id, amount_in_usdc, prix_moment_achat):
    global sell_profit_target, stop_loss_threshold
    log_message(f"Monitoring position for {product_id} with TP: {sell_profit_target}, SL: {stop_loss_threshold}")
    save_logs_to_file()

    while True:
        try:
            log_message(f"recuperons le prix actuel de {product_id}")
            save_logs_to_file()
            prix_actuel = get_market_price(product_id)
            if prix_actuel is None:
                log_message(f"Échec de la récupération du prix actuel pour {product_id}, nous réessayons...")
                save_logs_to_file()
                time.sleep(5)
                continue
            #log_message(f"Le prix actuel de {product_id} est : {prix_actuel} USDC")
            log_message(f"Le prix au moment de l'achat du {product_id} était : {prix_moment_achat} USDC")
            save_logs_to_file()
            objectif_takeprofit = prix_moment_achat + (prix_moment_achat * sell_profit_target)
            objectif_stoploss = prix_moment_achat - (prix_moment_achat * stop_loss_threshold)
            log_message(f"Take profit % : {sell_profit_target}")
            save_logs_to_file()
            log_message(f"Take profit objectif : {objectif_takeprofit} USDC")
            save_logs_to_file()
            log_message(f"Stop Loss % : {stop_loss_threshold}")
            save_logs_to_file()
            log_message(f"Stop Loss objectif : {objectif_stoploss} USDC")
            save_logs_to_file()
            log_message(f"vérifions les conditions de take profit et de stop loss...")
            save_logs_to_file()

            profit = (prix_moment_achat * sell_profit_target)
            if prix_actuel >= prix_moment_achat + profit:
                log_message(f"Take profit atteint pour {product_id} au prix de {prix_actuel}. Nous procedons à la vente de nos {product_id}.")
                save_logs_to_file()
                log_message(f"Vendons au montant acheté soit {prix_moment_achat} et gardons le profit {profit} {product_id} moins les frais de transactions")
                save_logs_to_file()
                place_market_sell(product_id, amount_in_usdc, prix_moment_achat)
                log_message(f"=====================================================================")
                save_logs_to_file()
                break
            # Check stop loss condition
            stoploss = (prix_moment_achat * stop_loss_threshold)
            if prix_actuel <= prix_moment_achat - stoploss:
                log_message(f"Stop loss atteint pour {product_id} au prix de {prix_actuel}. Nous procedons à la vente de nos {product_id}.")
                save_logs_to_file()
                log_message(f"Vendons au montant acheté soit {prix_moment_achat} et contentonnons de perdre {stoploss} {product_id} plus les frais de transactions")
                save_logs_to_file()
                place_market_sell(product_id, amount_in_usdc, prix_moment_achat)
                log_message(f"=====================================================================")
                save_logs_to_file()
                break
            log_message(f"Les conditions de take profit et de stop loss ne sont pas remplit...")
            save_logs_to_file()
            log_message(f"=====================================================================")
            save_logs_to_file()
            time.sleep(1)
            if not bot_running:
                log_message(f"DCA trading bot stopped")
                save_logs_to_file()
                break
        except Exception as e:
            log_message(f"Error monitoring position for TP/SL: {e}")
            save_logs_to_file()
            time.sleep(10)
#####################################################################################################
#VALIDE
def place_market_sell(product_id, amount_in_usdc,prix_moment_achat ):
    """Place a market sell order ensuring the order size meets Coinbase's requirements."""
    try:
############################
        #le prix actuelle
        #price=get_market_price(product_id)
        amount_in_btc = (1/prix_moment_achat) * amount_in_usdc


        # Fetch precision requirements for the base currency (BTC)
        product_details = client.get_product(product_id)
        base_increment = product_details['base_increment']
        log_message(f"{product_id} base increment is: {base_increment}")
        save_logs_to_file()
        # Validate and calculate precision
        precision = base_increment.find('1')
        if precision == -1:
            raise ValueError(f"Invalid base_increment format: {base_increment}")
        precision -= 1

        # Apply rounding to match the precision level expected by Coinbase
        amount_in_btc1 = amount_in_btc.quantize(Decimal('1.' + '0' * precision), rounding=ROUND_DOWN)

############################
        # Log the adjusted base currency amount
        log_message(f"Adjusted {product_id} amount with precision for sell: {amount_in_btc1}")
        save_logs_to_file()

        # Define the required individual arguments for create_order
        client_order_id = str(uuid.uuid4())  # Generate a unique client order ID
        side = "SELL"
        order_configuration = {
            "market_market_ioc": {
                "base_size": str(amount_in_btc1)  # Specify in base currency (BTC)
            }
        }

        # Place the order with required arguments
        response = client.create_order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=side,
            order_configuration=order_configuration
        )
        log_message(f"Market sell order response for {product_id}: {response}")
        save_logs_to_file()
        return response
    except KeyError as ke:
        log_message(f"Missing expected key in product details: {ke}")
        save_logs_to_file()
    except ValueError as ve:
        log_message(f"Invalid value encountered: {ve}")
        save_logs_to_file()
    except Exception as e:
        log_message(f"Error placing market sell order for {product_id}: {e}")
        save_logs_to_file()
    return None

#####################################################################################################
#VALIDE
def get_position_value(selected_crypto_pair):
    """Calculate the current USD value of the crypto holdings."""
    balance = get_account_balance(selected_crypto_pair)
    market_price = get_market_price(selected_crypto_pair)
    if balance and market_price:
        return balance * market_price
    return None
####################################################################################################################
def place_market_sell2(product_id, amount_in_usdc):
    """Place a market sell order ensuring the order size meets Coinbase's requirements."""
    try:
############################
        # Fetch precision requirements for the base currency (BTC)
        product_details = client.get_product(product_id)
        base_increment = product_details['base_increment']
        log_message(f"{product_id} base increment is: {base_increment}")
        save_logs_to_file()
        # Validate and calculate precision
        precision = base_increment.find('1')
        if precision == -1:
            raise ValueError(f"Invalid base_increment format: {base_increment}")
        precision -= 1

        # Apply rounding to match the precision level expected by Coinbase
        amount_in_btc1 = amount_in_usdc.quantize(Decimal('1.' + '0' * precision), rounding=ROUND_DOWN)

############################
        # Log the adjusted base currency amount
        #log_message(f"Adjusted {product_id} amount with precision for sell: {amount_in_btc1}")
        save_logs_to_file()

        # Define the required individual arguments for create_order
        client_order_id = str(uuid.uuid4())  # Generate a unique client order ID
        side = "SELL"
        order_configuration = {
            "market_market_ioc": {
                "base_size": str(amount_in_btc1)  # Specify in base currency (BTC)
            }
        }

        # Place the order with required arguments
        response = client.create_order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=side,
            order_configuration=order_configuration
        )
        log_message(f"Market sell order response for {product_id}: {response}")
        save_logs_to_file()
        return response
    except KeyError as ke:
        log_message(f"Missing expected key in product details: {ke}")
        save_logs_to_file()
    except ValueError as ve:
        log_message(f"Invalid value encountered: {ve}")
        save_logs_to_file()
    except Exception as e:
        log_message(f"Error placing market sell order for {product_id}: {e}")
        save_logs_to_file()
    return None

def remove_last_char_if_in_list(string, predefined_list):
    if string and string[-1] in predefined_list:
        return string[:-1]  # Supprime le dernier caractère
    return string  # Retourne la chaîne telle quelle si le caractère n'est pas dans la liste

def convert_to_usdc(account, selected_crypto_base):
    try:
        # Vérifier si le compte a des fonds
        if Decimal(account['available_balance']['value']) > 0:
            log_message(f"Le compte {account['name']} a des fonds : {account['available_balance']['value']} {account['available_balance']['currency']}")
            save_logs_to_file()
            currency = account['available_balance']['currency']
            # Effectuer la conversion en USDC
            if currency != 'USDC' and currency != selected_crypto_base:
                conversion_amount = Decimal(account['available_balance']['value'])
                log_message(f"Conversion de {conversion_amount} {account['available_balance']['currency']} en USDC...")
                save_logs_to_file()
                #netoyer le nom du porteuille si il contient un chiffre à la fin de son nom exemple de ETH2
                # Exemple d'utilisation
                predefined_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                newcurrency = remove_last_char_if_in_list(currency, predefined_list)
                to_account = 'USDC'
                product_id = newcurrency+'-'+to_account
                place_market_sell2(product_id, conversion_amount)
                return True  # Simule que la conversion est réussie
            else:
                log_message(f"Le compte {account['name']} ne peut pas être débité car il fait partie des crypto qui forment la paire {selected_crypto_base}-USDC.")
                save_logs_to_file()
                return False
        else:
            log_message(f"Le compte {account['name']} n'a pas de fonds.")
            save_logs_to_file()
            return False
    except Exception as e:
        log_message(f"Erreur lors de la vérification des fonds du compte {account['name']} : {e}")
        save_logs_to_file()
        return False

def check_and_convert_all_accounts(selected_crypto_base):
    global accounts
    try:
        # Récupérer tous les comptes
        #accounts = client.get_accounts()
        log_message("Analyse des comptes en cours...")
        save_logs_to_file()
        # Parcourir tous les comptes et vérifier s'il y a des fonds
        for account in accounts['accounts']:
            convert_to_usdc(account, selected_crypto_base)
    except Exception as e:
        log_message(f"Erreur lors de la récupération des comptes : {e}")
        save_logs_to_file()
#####################################################################################################
#VERIFIE REST A TESTER
def dca_trading_bot():
    """DCA trading bot with automated buy based on percentages."""
    global bot_running, buy_percentage_of_capital
    bot_running = True
    stopping = False  # Indique si un arrêt est en cours
    log_message("DCA trading bot started")
    save_logs_to_file()

    while bot_running or stopping:  # S'assurer que les processus en cours sont terminés
        try:
            # Pour chaque paire de crypto-monnaies sélectionnée
            for selected_crypto_pair in selected_crypto_pairs:
                # Si un arrêt est demandé, sortir de la boucle principale
                if not bot_running:
                    log_message("Arrêt demandé. Finalisation des processus en cours.")
                    save_logs_to_file()
                    stopping = True
                    break  # Quitter la boucle des paires pour arrêter proprement

                # Identité de la paire traitée
                product_id = selected_crypto_pair
                log_message(f"Paire traitée actuellement : {product_id}")
                save_logs_to_file()

                # Vérification du solde USDC
                usdc_balance = get_usdc_balance()
                log_message(f"Le solde USDC est : {usdc_balance}")
                save_logs_to_file()

                # Déterminer le montant à acheter
                buy_amount = usdc_balance * buy_percentage_of_capital
                if usdc_balance <= Decimal('1.0'):
                    log_message(f"Solde USDC insuffisant pour placer un ordre d'achat de : {product_id}.")
                    save_logs_to_file()
                    continue  # Passer à la paire suivante

                # Achat avec ou sans IA
                if ia:
                    log_message("IA activated.")
                    save_logs_to_file()
                    trend_comparison = will_crypto_increase_or_decrease(yesterday_last_real, today_pred)
                    if trend_comparison > 0:
                        log_message(f"{product_id} Prendra de la valeur, achat en cours.")
                        save_logs_to_file()
                        place_market_buy(product_id)
                    else:
                        log_message(f"{product_id} Perdra de la valeur, achat annulé.")
                        save_logs_to_file()
                else:
                    log_message(f"Placons un ordre d'achat d'un montant de {buy_amount} pour : {product_id}.")
                    save_logs_to_file()
                    place_market_buy(product_id)

            # Mise en pause après avoir traité toutes les paires
            log_message("Toutes les paires traitées. Mise en pause du robot.")
            save_logs_to_file()
            time.sleep(3)  # Pause avant de recommencer la boucle

        except Exception as e:
            log_message(f"Exception in DCA trading bot: {e}")
            save_logs_to_file()
            time.sleep(15)

    log_message("Finalisation des processus terminée. Arrêt du bot.")
    save_logs_to_file()
#derniere version prenant en compte l ia
#####################################################################################################
# Fonction pour vérifier et comparer les soldes toutes les secondes
def Balance_Total():
    global log_data1  # Utiliser la variable soldes initiaux définie en dehors de la fonction

    while True:
        # Réinitialiser log_data à chaque itération avant d'ajouter de nouveaux logs
        log_data1 = ""  # Effacer les logs précédents

        # Assurez-vous que vous obtenez bien les informations de solde et les manipulez correctement
        try:
            # Récupérer les portefeuilles
            transactions = client.get_transaction_summary()
            balance_total = transactions['total_balance']
            log_data1 += f"{balance_total}\n"

            accounts = client.get_accounts()  # Obtenez les comptes
            print("mise à jour des comptes")
        except KeyError as e:
            log_data1 += f"Erreur de récupération de la balance: {str(e)}\n"

        # Envoyer les données mises à jour au client via SocketIO
        #socketio.emit('update_log1', {'log_Balance_Total': log_data1})

        # Attendre une seconde avant de vérifier à nouveau
        time.sleep(1)

# Créer et démarrer le thread
thread = threading.Thread(target=Balance_Total)
thread.daemon = True  # Ensure the thread exits when the main program exits
thread.start()
#####################################################################################
#########################################################################################
accounts = client.get_accounts()
#########################################################################################
def get_usdc_balance():
    global accounts
    try:
        #accounts = client.get_accounts()
        for account in accounts['accounts']:
            if account['currency'] == 'USDC':
                return Decimal(account['available_balance']['value'])
    except Exception as e:
        log_message(f"Error fetching USDC balance: {e}")
        save_logs_to_file()
    return Decimal('0')
#########################################################################################
#########################################################################################
def get_eth2_balance():
    global accounts
    try:
        #accounts = client.get_accounts()
        for account in accounts['accounts']:
            if account['currency'] == 'BTC':
                return Decimal(account['available_balance']['value'])
    except Exception as e:
        log_message(f"Error fetching BTC balance: {e}")
        save_logs_to_file()
    return Decimal('0')
#########################################################################################
# Fonction pour vérifier et comparer les soldes toutes les secondes
def Your_Usdc():
    global log_data2  # Utiliser la variable soldes initiaux définie en dehors de la fonction

    while True:
        # Réinitialiser log_data à chaque itération avant d'ajouter de nouveaux logs
        log_data2 = ""  # Effacer les logs précédents

        # Assurez-vous que vous obtenez bien les informations de solde et les manipulez correctement
        try:
            # Récupérer les portefeuilles
            usdc_balance = get_usdc_balance()
            log_data2 += f"{usdc_balance:.6f}\n"
        except KeyError as e:
            log_data2 += f"Erreur de récupération de la balance: {str(e)}\n"

        # Envoyer les données mises à jour au client via SocketIO
        #socketio.emit('update_log2', {'log_usdc_balance': log_data2})

        # Attendre une seconde avant de vérifier à nouveau
        time.sleep(1.6)

# Créer et démarrer le thread
thread1 = threading.Thread(target=Your_Usdc)
thread1.daemon = True  # Ensure the thread exits when the main program exits
thread1.start()
#####################################################################################
#########################################################################################
# Fonction pour vérifier et comparer les soldes toutes les secondes
def Your_Eth2():
    global log_data3  # Utiliser la variable soldes initiaux définie en dehors de la fonction

    while True:
        # Réinitialiser log_data à chaque itération avant d'ajouter de nouveaux logs
        log_data3 = ""  # Effacer les logs précédents

        # Assurez-vous que vous obtenez bien les informations de solde et les manipulez correctement
        try:
            # Récupérer les portefeuilles
            eth2_balance = get_eth2_balance()
            log_data3 += f"{eth2_balance:.6f}\n"
        except KeyError as e:
            log_data3 += f"Erreur de récupération de la balance: {str(e)}\n"

        # Envoyer les données mises à jour au client via SocketIO
        #socketio.emit('update_log3', {'log_eth2_balance': log_data3})

        # Attendre une seconde avant de vérifier à nouveau
        time.sleep(1.7)

# Créer et démarrer le thread
thread2 = threading.Thread(target=Your_Eth2)
thread2.daemon = True  # Ensure the thread exits when the main program exits
thread2.start()
#####################################################################################
#####################################################################################
# Fonction pour récupérer les soldes initiaux (une fois par jour)
def get_soldes_initiaux():
    soldes_initiaux = {}
    global log_data
    for account in accounts.accounts:
        solde_initial = float(account.available_balance['value'])
        currency = account.available_balance['currency']
        soldes_initiaux[account] = (solde_initial, currency)
        log_data += f"Solde initial pour le compte {currency}: {solde_initial} {currency}\n"
    return soldes_initiaux

# Récupérer les soldes initiaux pour commencer
soldes_initiaux = get_soldes_initiaux()
#####################################################################################
# Fonction pour obtenir la valeur en temps réel d'une cryptomonnaie via l'API Coinbase
def get_crypto_value(crypto_pair):
    url = f"https://api.coinbase.com/v2/prices/{crypto_pair}/buy"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie si la requête a échoué (code HTTP 4xx ou 5xx)

        # Tentons de décoder le JSON
        try:
            data = response.json()
            # Vérification si la structure attendue est présente
            if 'data' in data and 'amount' in data['data']:
                return float(data['data']['amount'])
            else:
                raise ValueError("Réponse invalide: 'data' ou 'amount' manquants.")
        except ValueError as e:
            raise ValueError(f"Erreur lors de l'analyse de la réponse JSON: {e}")

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Erreur lors de la requête à l'API Coinbase: {e}")
    except Exception as e:
        raise Exception(f"Erreur dans get_crypto_value pour {crypto_pair}: {e}")
#####################################################################################
# Fonction pour vérifier et comparer les soldes toutes les secondes
def check_soldes():
    global soldes_initiaux, log_data, Profit_cumul, total, accounts  # Utiliser les variables globales nécessaires

    while True:
        try:
            # voir si ca fonctionne
            Profit_cumul = 0

            log_data = ""  # Réinitialiser les logs
            heure_locale = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_data += f"Dernière mise à jour : {heure_locale}\n"

            for account in accounts.accounts:
                solde_initial, currency = soldes_initiaux.get(account, (0, 'USD'))  # Valeur par défaut si non trouvé
                try:
                    crypto = account.available_balance['currency']
                    accountts = client.get_accounts()
                    for accountt in accountts['accounts']:
                        if accountt['currency'] == crypto:
                            solde_actuel = float(accountt['available_balance']['value'])
                    #solde_actuel = float(account.available_balance['value'])
                    log_data += f"------------------------------------------------\n"
                    log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"
                    log_data += f"PORTEFEUILLE {crypto}\n"
                    log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"
                    log_data += f"Solde initial : {solde_initial} {crypto}\n"
                    log_data += f"Solde actuel  : {solde_actuel} {crypto}\n"

                    # Calculer la différence entre le solde initial et le solde actuel
                    difference = solde_actuel - solde_initial
                    log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"
                    log_data += f"Profit du jour pour le compte {currency}: {difference:.2f} {currency}\n"

                    # Récupérer la valeur en USD
                    crypto_pair = crypto + "-USD"
                    try:
                        value_in_usd = get_crypto_value(crypto_pair)
                        log_data += f"La valeur de {crypto} en USD est : {value_in_usd}\n"
                        total = value_in_usd * difference
                        log_data += f"Conversion de vos bénéfices {crypto} en USD = {total:.2f} USD\n"
                        Profit_cumul += total
                    except Exception as e:
                        log_data += f"Erreur lors de la récupération de la valeur de {crypto} en USD : {str(e)}\n"
                        continue  # Passer à la paire suivante en cas d'erreur

                    log_data += "::::::::::::::::::::::::::::::::::::::::::::::::\n"

                    # Vérifier si la date a changé (si un nouveau jour commence)
                    current_time = datetime.now()
                    if current_time.hour == 0 and current_time.minute == 0:  # Si c'est minuit
                        log_data += "Mise à jour des soldes initiaux pour le nouveau jour...\n"
                        soldes_initiaux = get_soldes_initiaux()

                except Exception as e:
                    log_data += f"Erreur avec le portefeuille {crypto}: {str(e)}\n"
                    continue  # Passer au compte suivant

            log_data += f"PROFIT CUMULE : {Profit_cumul:.2f} USD\n"

            # Envoyer les données mises à jour au client
            #socketio.emit('update_log', {'log': log_data})

        except Exception as e:
            # Enregistrer toute autre erreur non prévue
            log_data += f"Erreur générale dans le thread check_soldes : {str(e)}\n"

        finally:
            # Toujours attendre avant de recommencer pour éviter une surcharge
            time.sleep(4.5)

# Créer et démarrer le thread
thread3 = threading.Thread(target=check_soldes)
thread3.daemon = True  # Assure que le thread s'arrête lorsque le programme principal s'arrête
thread3.start()
#########################################################################################
# Fonction pour vérifier et comparer les ordres toutes les secondes
def les_ordres():
    global log_data4  # Utilisation de la variable globale log_data
    while True:
        # Réinitialiser log_data à chaque itération
        log_data4 = ""
        try:
            heure_locale = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_data4 += f"Dernière mise à jour : {heure_locale}\n"
            # Obtenir tous les ordres
            orders_data = client.list_orders()
            # Conversion de la chaîne JSON en dictionnaire Python
            orders_dict = orders_data

            # Parcourir et traiter les données des commandes
            #for order in orders_dict['orders']:
            for order in orders_dict['orders'][:30]:
                order_id = order['order_id']
                product_id = order['product_id']
                user_id = order['user_id']
                side = order['side']
                client_order_id = order['client_order_id']
                order_status = order['status']
                time_in_force = order['time_in_force']
                created_time = order['created_time']
                completion_percentage = order['completion_percentage']
                filled_size = order['filled_size']
                average_filled_price = order['average_filled_price']
                fee = order['fee']
                number_of_fills = order['number_of_fills']
                filled_value = order['filled_value']
                pending_cancel = order['pending_cancel']
                size_in_quote = order['size_in_quote']
                total_fees = order['total_fees']
                size_inclusive_of_fees = order['size_inclusive_of_fees']
                total_value_after_fees = order['total_value_after_fees']
                trigger_status = order['trigger_status']
                order_type = order['order_type']
                reject_reason = order['reject_reason']
                settled = order['settled']
                product_type = order['product_type']
                reject_message = order['reject_message']
                cancel_message = order['cancel_message']
                order_placement_source = order['order_placement_source']
                outstanding_hold_amount = order['outstanding_hold_amount']
                is_liquidation = order['is_liquidation']
                last_fill_time = order['last_fill_time']
                edit_history = order['edit_history']
                leverage = order['leverage']
                margin_type = order['margin_type']
                retail_portfolio_id = order['retail_portfolio_id']
                originating_order_id = order['originating_order_id']
                attached_order_id = order['attached_order_id']
                attached_order_configuration = order['attached_order_configuration']
                #################################
                # Ajouter les informations de l'ordre au log
                log_data4 += f"------------------------------------------------\n"
                log_data4 += f"Order ID: {order_id}\n"
                log_data4 += f"Product ID: {product_id}\n"
                log_data4 += f"User ID: {user_id}\n"
                log_data4 += f"side: {side}\n"
                log_data4 += f"client_order_id: {client_order_id}\n"
                log_data4 += f"Status: {order_status}\n"
                log_data4 += f"time_in_force: {time_in_force}\n"
                log_data4 += f"created_time: {created_time}\n"
                log_data4 += f"completion_percentage: {completion_percentage}\n"
                log_data4 += f"Filled Size: {filled_size}\n"
                log_data4 += f"Average Filled Price: {average_filled_price}\n"
                log_data4 += f"fee: {fee}\n"
                log_data4 += f"number_of_fills: {number_of_fills}\n"
                log_data4 += f"filled_value: {filled_value}\n"
                log_data4 += f"pending_cancel: {pending_cancel}\n"
                log_data4 += f"size_in_quote: {size_in_quote}\n"
                log_data4 += f"Total Fees: {total_fees}\n"
                log_data4 += f"size_inclusive_of_fees: {size_inclusive_of_fees}\n"
                log_data4 += f"total_value_after_fees: {total_value_after_fees}\n"
                log_data4 += f"trigger_status: {trigger_status}\n"
                log_data4 += f"order_type: {order_type}\n"
                log_data4 += f"reject_reason: {reject_reason}\n"
                log_data4 += f"settled: {settled}\n"
                log_data4 += f"product_type: {product_type}\n"
                log_data4 += f"reject_message: {reject_message}\n"
                log_data4 += f"cancel_message: {cancel_message}\n"
                log_data4 += f"order_placement_source: {order_placement_source}\n"
                log_data4 += f"outstanding_hold_amount: {outstanding_hold_amount}\n"
                log_data4 += f"is_liquidation: {is_liquidation}\n"
                log_data4 += f"last_fill_time: {last_fill_time}\n"
                log_data4 += f"edit_history: {edit_history}\n"
                log_data4 += f"leverage: {leverage}\n"
                log_data4 += f"margin_type: {margin_type}\n"
                log_data4 += f"retail_portfolio_id: {retail_portfolio_id}\n"
                log_data4 += f"originating_order_id: {originating_order_id}\n"
                log_data4 += f"attached_order_id: {attached_order_id}\n"
                log_data4 += f"attached_order_configuration: {attached_order_configuration}\n"
                #################################

        except Exception as e:
            # Gestion des exceptions et ajout d'un message d'erreur aux logs
            log_data4 += f"Erreur lors de la récupération des ordres : {str(e)}\n"

        # Envoyer les données mises à jour au client via SocketIO
        #socketio.emit('update_log4', {'log_orders': log_data4})

        # Pause d'une seconde avant de recommencer
        time.sleep(2.5)

# Créer et démarrer le thread
thread4 = threading.Thread(target=les_ordres)
thread4.daemon = True  # Ensure the thread exits when the main program exits
thread4.start()
#####################################################################################
#####################################################################################################
def send_2fa_code():
    global current_2fa_code
    current_2fa_code = totp.now()  # Generate the 2FA code

    # Create and send the email with the 2FA code
    subject = "Your 2FA Code"
    body = f"Your 2FA code is: {current_2fa_code}"

    msg = Message(subject, recipients=[user_email])
    msg.body = body

    try:
        with mail.connect() as connection:  # Explicitly connect to SMTP server
            connection.send(msg)
        log_message(f"Sent 2FA code to {user_email}")
        save_logs_to_file()
    except Exception as e:
        log_message(f"Error sending 2FA code: {e}")
        save_logs_to_file()
######################################################################
def send_failed_login_alert():

    # Vérification de la variable user_email
    if not user_email:
        print("Error: User email is not set.")
        return  # Retourne sans envoyer l'email si l'email utilisateur n'est pas défini

    # Définir le sujet et le corps de l'email
    subject = "Failed Login Attempt"
    body = "Une tentative de connexion a échoué."

    # Créer le message email
    msg = Message(subject, recipients=[user_email])
    msg.body = body

    try:
        print(f"Attempting to send email to {user_email}")  # Vérifier si l'email est bien envoyé
        # Tenter d'envoyer l'email en utilisant la connexion SMTP
        with mail.connect() as connection:  # Connexion explicite au serveur SMTP
            connection.send(msg)
        log_message(f"Sent failed login alert to {user_email}")  # Si l'email est envoyé avec succès
        save_logs_to_file()
    except Exception as e:
        log_message(f"Error sending failed login alert: {str(e)}")  # Log de l'erreur si l'envoi échoue
        save_logs_to_file()
        print(f"Error sending failed login alert: {str(e)}")  # Affichage de l'erreur pour le débogage
#########################################################################################
# Decorator to require login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function
#####################################################################################
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == HARDCODED_PASSWORD:
            send_2fa_code()  # Send 2FA code to the user
            return render_template('verify_2fa.html')  # Show the 2FA verification form
        else:
            send_failed_login_alert()
            return render_template('login.html', error="Incorrect password")

    return render_template('login.html')
#####################################################################################
@app.route('/verify_2fa', methods=['POST'])
def verify_2fa():
    entered_2fa_code = request.form.get('2fa_code')
    if entered_2fa_code == current_2fa_code:
        session['logged_in'] = True
        return redirect(url_for('index'))
    else:
        return render_template('verify_2fa.html', error="Invalid 2FA code")
#####################################################################################
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))
####################################################################
# Protect the main route with login_required
@app.route('/')
@login_required
def index():
    form_data = {
        #"risk_level": "moderate",  # Default values for form data
        #"amount": 0.0001,
        #"compounding": False
        "ADA_USDC": True,
        "AAVE_USDC": True,
        "AERO_USDC": True,
        "ALGO_USDC": True,
        "AMP_USDC": True,
        "ARB_USDC": True,
        "AVAX_USDC": True,
        "BCH_USDC": True,
        "BONK_USDC": True,
        "BTC_USDC": True,
        "CRV_USDC": True,
        "DOGE_USDC": True,
        "DOT_USDC": True,
        "ETH_USDC": True,
        "EURC_USDC": True,
        "FET_USDC": True,
        "FIL_USDC": True,
        "GRT_USDC": True,
        "HBAR_USDC": True,
        "ICP_USDC": True,
        "IDEX_USDC": True,
        "INJ_USDC": True,
        "JASMY_USDC": True,
        "JTO_USDC": True,
        "LINK_USDC": True,
        "LTC_USDC": True,
        "MOG_USDC": True,
        "NEAR_USDC": True,
        "ONDO_USDC": True,
        "PEPE_USDC": True,
        "RENDER_USDC": True,
        "RNDR_USDC": True,
        "SEI_USDC": True,
        "SHIB_USDC": True,
        "SOL_USDC": True,
        "SUI_USDC": True,
        "SUPER_USDC": True,
        "SUSHI_USDC": True,
        "SWFTC_USDC": True,
        "TIA_USDC": True,
        "UNI_USDC": True,
        "USDT_USDC": True,
        "VET_USDC": True,
        "WIF_USDC": True,
        "XLM_USDC": True,
        "XYO_USDC": True,
        "XRP_USDC": True,
        "YFI_USDC": True,
        "ETC_USDC": True,
        "MATIC_USDC": True,
        "buy_percentage_of_capital": Decimal("0.05"),
        #"sell_percentage_of_capital": Decimal("0.05"),
        "sell_profit_target": Decimal("0.005"),
        "stop_loss_threshold": Decimal("0.005"),
        "dca_interval_minute": 1
    }
    return render_template('index.html', bot_running=bot_running, form_data=form_data, logs="\n".join(logs[-100:]),log_Balance_Total=log_data1, log=log_data, log_usdc_balance=log_data2, log_eth2_balance=log_data3, log_orders=log_data4)
    #return render_template('index.html', bot_running=bot_running, form_data=form_data,logs="\n".join(logs[-100:]))
####################################################################
@app.route('/start', methods=['POST'])
def start_bot():
    global bot_running
    if not bot_running:
        bot_thread = Thread(target=dca_trading_bot)
        bot_thread.daemon = True
        bot_thread.start()
    return redirect(url_for('index'))
#####################################################################################
@app.route('/stop', methods=['POST'])
def stop_bot():
    global bot_running
    bot_running = False
    return redirect(url_for('index'))
####################################################################
@app.route('/update_settings', methods=['POST'])
def update_settings():
    global selected_crypto_pairs, buy_percentage_of_capital, sell_profit_target, stop_loss_threshold, dca_interval_minute,ia
    #user_risk_level = request.form.get('risk_level', user_risk_level)
    #compounding_enabled = request.form.get('compounding') == 'on'
    #initial_trade_amount = float(request.form.get('amount', initial_trade_amount))
    buy_percentage_of_capital = Decimal(request.form.get('buy_percentage_of_capital', buy_percentage_of_capital))
    #sell_percentage_of_capital = Decimal(request.form.get('sell_percentage_of_capital', sell_percentage_of_capital))
    sell_profit_target = Decimal(request.form.get('sell_profit_target', sell_profit_target))
    stop_loss_threshold = Decimal(request.form.get('stop_loss_threshold', stop_loss_threshold))
    dca_interval_minute = int(request.form.get('dca_interval_minute', dca_interval_minute))
    selected_crypto_pairs = request.form.getlist('selected_crypto_pairs')
    ia = request.form.getlist('ia')
    # selected_crypto_pair = request.form.get('crypto_pair', 'BTC-USDC')
    log_message(f"Settings updated: selected_crypto_pairs={selected_crypto_pairs},buy_percentage_of_capital={buy_percentage_of_capital}, sell_profit_target={sell_profit_target}, stop_loss_threshold={stop_loss_threshold}, dca_interval_minute={dca_interval_minute}, ia={ia}")
    save_logs_to_file()
    return redirect(url_for('index'))
####################################################################
@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify({"logs": logs[-100:]})  # Send logs as an array of strings
#######################################################
@app.route('/log_Balance_Total', methods=['GET'])
def log_Balance_Total():
    return jsonify({"log_Balance_Total": log_data1})  # Send logs as an array of strings

@app.route('/log_usdc_balance', methods=['GET'])
def log_usdc_balance():
    return jsonify({"log_usdc_balance": log_data2})  # Send logs as an array of strings

@app.route('/log_eth2_balance', methods=['GET'])
def log_eth2_balance():
    return jsonify({"log_eth2_balance": log_data3})  # Send logs as an array of strings

@app.route('/log_orders', methods=['GET'])
def log_orders():
    return jsonify({"log_orders": log_data4})  # Send logs as an array of strings

@app.route('/log', methods=['GET'])
def log():
    return jsonify({"log": log_data})  # Send logs as an array of strings

#######################################################
# API pour obtenir les données des prédictions en temps réel
@app.route('/get_predictions')
def get_predictions():
    return jsonify({'predictions': {crypto: all_predictions[crypto].tolist() for crypto in all_predictions}})


if __name__ == '__main__':
    socketio.run(app, debug=False)