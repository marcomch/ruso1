import time
import logging
from iqoptionapi.stable_api import IQ_Option
import datetime
import sys
from threading import Timer
from colorama import init, Fore, Back
from configobj import ConfigObj
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

init(autoreset=True)
green = Fore.GREEN
yellow = Fore.YELLOW
red = Fore.RED
white = Fore.WHITE
greenf = Back.GREEN
yellowf = Back.YELLOW
redf = Back.RED
blue = Fore.BLUE

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lucro_total = 0
total_wins = 0
total_losses = 0
total_ties = 0

try:
    model = joblib.load('digit_classifier.pkl')
    le_color = joblib.load('le_color.pkl')
    le_decision = joblib.load('le_decision.pkl')
    le_result = joblib.load('le_result.pkl')
    logging.info("Modelo y codificadores cargados correctamente.")
except FileNotFoundError:
    model = None
    le_color = None
    le_decision = None
    le_result = None
    logging.warning("Modelo o codificadores no encontrados. La IA no estará activa.")

def registrar_operacion(asset, action, resultado, lucro, digits_info, decision, pred_result=None, pred_prob=None):
    digits_str = '; '.join([f"Tick {i+1}: Dígito={info['digit']} ({info['color']})" for i, info in enumerate(digits_info)])
    pred_str = f" | Predicción IA: {pred_result} ({pred_prob:.2f})" if pred_result and pred_prob else ""
    with open('historico_operaciones.txt', 'a') as archivo:
        archivo.write(
            f'{datetime.datetime.now()} | Activo: {asset} | Dirección: {action} | '
            f'Resultado: {resultado} | Lucro/Pérdida: {lucro} | '
            f'Análisis: [{digits_str}] | Decisión: {decision}{pred_str}\n'
        )

class NumberPressureBot:
    def __init__(self, email, password, valor_entrada, asset, account_type="PRACTICE", api=None):
        self.email = email
        self.password = password
        self.api = api
        self.asset = self._validate_asset(asset)
        self.account_type = account_type
        self.ticks = []
        self.max_ticks = 10
        self.analyze_ticks = 4
        self.candle_duration = 60
        self.trade_amount = float(valor_entrada)
        self.expiration_mode = 1
        self.check_time = 59
        self.tick_start_time = 40
        self.win_threshold = 0.6

    def _validate_asset(self, asset):
        if not self.api:
            raise ValueError("API no está inicializada. Conéctese primero.")
        
        asset = asset.strip().upper()
        activos = self.api.get_all_ACTIVES_OPCODE()
        valid_assets = [activo.upper() for activo in activos.keys()]
        
        if asset not in valid_assets:
            suggested_asset = 'USDZAR-OTC' if asset == 'USZAR-OTC' else None
            error_msg = f"Activo inválido: {asset}. Ejemplos de activos válidos: USDZAR-OTC, EURJPY-OTC, EURUSD-OTC."
            if suggested_asset:
                error_msg += f" ¿Querías decir {suggested_asset}?"
            raise ValueError(error_msg)
        return asset

    def connect(self):
        if not self.api:
            self.api = IQ_Option(self.email, self.password)
        check, reason = self.api.connect()
        if check:
            logging.info("Conexión exitosa.")
            self.api.change_balance(self.account_type)
            logging.info(f"Cuenta seleccionada: {self.account_type}")
            return True
        else:
            logging.error(f"Conexión fallida: {reason}")
            return False

    def check_connect(self):
        if not self.api.check_connect():
            logging.info("Intentando reconectar...")
            check, reason = self.api.connect()
            if check:
                logging.info("Reconexión exitosa.")
                self.api.change_balance(self.account_type)
                logging.info(f"Cuenta re-seleccionada: {self.account_type}")
                return True
            else:
                logging.error(f"Reconexión fallida: {reason}")
                return False
        return True

    def sincronizar_activos(self):
        logging.info("Sincronizando activos con IQ Option...")
        self.api.update_ACTIVES_OPCODE()
        activos = self.api.get_all_ACTIVES_OPCODE()
        logging.info("Activos sincronizados correctamente.")
        instrument_type = "binary-option"
        self.api.subscribe_top_assets_updated(instrument_type)
        logging.info("Esperando la actualización de activos...")
        return activos

    def get_last_digit(self, price):
        price_str = str(price)
        if len(price_str) >= 2 and '.' in price_str:
            decimal_part = price_str.split('.')[-1]
            if len(decimal_part) >= 2:
                return int(decimal_part[-2])
            elif len(decimal_part) == 1:
                return int(decimal_part[-1])
        return 0

    def is_odd(self, digit):
        return digit % 2 != 0

    def predict_result(self, digits_info, decision):
        if model is None:
            return None, None

        try:
            data = {
                'digit1': digits_info[0]['digit'], 'color1': le_color.transform([digits_info[0]['color']])[0],
                'digit2': digits_info[1]['digit'], 'color2': le_color.transform([digits_info[1]['color']])[0],
                'digit3': digits_info[2]['digit'], 'color3': le_color.transform([digits_info[2]['color']])[0],
                'digit4': digits_info[3]['digit'], 'color4': le_color.transform([digits_info[3]['color']])[0],
                'decision': le_decision.transform([decision])[0]
            }
            df = pd.DataFrame([data])

            pred = model.predict(df)[0]
            pred_proba = model.predict_proba(df)[0]
            pred_result = le_result.inverse_transform([pred])[0]
            win_prob = pred_proba[le_result.transform(['WIN'])[0]]

            return pred_result, win_prob
        except Exception as e:
            logging.error(f"Error en la predicción: {e}")
            return None, None

    def analyze_pressure(self):
        if len(self.ticks) < self.max_ticks:
            print(f"{yellow} >> No hay suficientes ticks para el análisis ({len(self.ticks)}/{self.max_ticks})")
            return None, [], "No se realiza operación"

        recent_ticks = self.ticks[-self.analyze_ticks:]
        if len(recent_ticks) < self.analyze_ticks:
            print(f"{yellow} >> No hay suficientes ticks recientes para el análisis ({len(recent_ticks)}/{self.analyze_ticks})")
            return None, [], "No se realiza operación"

        digits = [self.get_last_digit(tick['price']) for tick in recent_ticks]
        colors = [tick['color'] for tick in recent_ticks]
        digits_info = [{'digit': digit, 'color': color} for digit, color in zip(digits, colors)]

        print(f"{blue}{'=' * 30}")
        print(f"{yellow}Análisis (Segundo {self.check_time}):")
        for i, (digit, color) in enumerate(zip(digits, colors)):
            color_text = blue if color == 'blue' else red
            print(f"{white}Tick {i+1}: Penúltimo Dígito={color_text}{digit}{white}")

        max_digit = max(digits)
        max_index = digits.index(max_digit)
        max_color = colors[max_index]
        max_is_odd = self.is_odd(max_digit)

        display_digit = max_digit
        if max_digit == 0:
            prev_digit = digits[max_index - 1] if max_index > 0 else digits[max_index + 1]
            display_digit = 10 if prev_digit > 5 else 0
            max_is_odd = False

        max_color_text = f"{blue}Azul" if max_color == 'blue' else f"{red}Rojo"
        parity_text = "Impar" if max_is_odd else "Par"
        print(f"{yellow}Penúltimo dígito más grande: {display_digit} ({parity_text}, {max_color_text})")

        interrupted = False
        interruption_details = []
        for i in range(max_index + 1, len(recent_ticks)):
            curr_digit = digits[i]
            curr_color = colors[i]
            curr_is_odd = self.is_odd(curr_digit)

            if curr_digit == 0:
                prev_digit = digits[i - 1] if i > 0 else digits[i + 1]
                curr_digit = 10 if prev_digit > 5 else 0
                curr_is_odd = False

            if max_color == 'red':
                if max_is_odd and curr_color == 'blue' and not curr_is_odd:
                    interrupted = True
                    interruption_details.append(f"Tick {i+1}: Par azul ({curr_digit}) interrumpe impar rojo")
                elif not max_is_odd and curr_color == 'blue' and curr_is_odd:
                    interrupted = True
                    interruption_details.append(f"Tick {i+1}: Impar azul ({curr_digit}) interrumpe par rojo")
            elif max_color == 'blue':
                if max_is_odd and curr_color == 'red' and not curr_is_odd:
                    interrupted = True
                    interruption_details.append(f"Tick {i+1}: Par rojo ({curr_digit}) interrumpe impar azul")
                elif not max_is_odd and curr_color == 'red' and curr_is_odd:
                    interrupted = True
                    interruption_details.append(f"Tick {i+1}: Impar rojo ({curr_digit}) interrumpe par azul")

        if interrupted:
            print(f"{red}Interrupción detectada:")
            for detail in interruption_details:
                print(f"{white}  - {detail}")
        else:
            print(f"{green}No hay interrupción.")

        if not interrupted:
            action = 'call' if max_color == 'blue' else 'put'
            action_text = "CALL (UP)" if action == 'call' else "PUT (DOWN)"
            print(f"{yellow}Decisión: {action_text}")
            return action, digits_info, action_text
        else:
            print(f"{yellow}Decisión: No se realiza operación.")
            return None, digits_info, "No se realiza operación"

    def place_trade(self, action, digits_info, decision, pred_result, pred_prob):
        global lucro_total, total_wins, total_losses, total_ties

        if not self.check_connect():
            return False

        check, order_id = self.api.buy(self.trade_amount, self.asset, action, self.expiration_mode)
        if check:
            logging.info(f"Operación {action} realizada con ID: {order_id}")
            print(f"{green}Operación {action.upper()} realizada con ID: {order_id}")

            while True:
                time.sleep(0.1)
                status, resultado = self.api.check_win_v4(order_id)
                if status:
                    lucro_total += round(resultado, 2)
                    if resultado > 0:
                        total_wins += 1
                        print(f"{green}>> Resultado: WIN \n>> Lucro: {round(resultado, 2)} \n>> Activo: {self.asset} \n>> Lucro total: {round(lucro_total, 2)}")
                        registrar_operacion(self.asset, action, 'WIN', round(resultado, 2), digits_info, decision, pred_result, pred_prob)
                    elif resultado < 0:
                        total_losses += 1
                        print(f"{red}>> Resultado: LOSS \n>> Pérdida: {round(resultado, 2)} \n>> Activo: {self.asset} \n>> Lucro total: {round(lucro_total, 2)}")
                        registrar_operacion(self.asset, action, 'LOSS', round(resultado, 2), digits_info, decision, pred_result, pred_prob)
                    else:
                        total_ties += 1
                        print(f"{yellow}>> Resultado: EMPATE \n>> Lucro: {round(resultado, 2)} \n>> Activo: {self.asset} \n>> Lucro total: {round(lucro_total, 2)}")
                        registrar_operacion(self.asset, action, 'EMPATE', round(resultado, 2), digits_info, decision, pred_result, pred_prob)
                    
                    print(f"{white}Recuento general - Wins: {total_wins}, Losses: {total_losses}, Ties: {total_ties}")
                    break
            return True
        else:
            logging.error("Fallo al realizar la operación")
            print(f"{red}Fallo al realizar la operación")
            registrar_operacion(self.asset, action, 'ERROR', 0, digits_info, decision, pred_result, pred_prob)
            return False

    def process_tick(self, tick_data):
        price = tick_data['price']
        color = 'blue' if len(self.ticks) == 0 or price >= self.ticks[-1]['price'] else 'red'
        self.ticks.append({'price': price, 'color': color})
        digit = self.get_last_digit(price)
        color_text = blue if color == 'blue' else red
        print(f"{white}T. recibido: Precio={color_text}{price}{white}, U. Dígito={color_text}{digit}{white}")
        if len(self.ticks) > self.max_ticks:
            self.ticks.pop(0)

    def run(self):
        if not self.connect():
            logging.error("No se pudo iniciar el bot debido a problemas de conexión.")
            print(f"{red}No se pudo iniciar el bot debido a problemas de conexión.")
            return

        activos = self.sincronizar_activos()
        self.api.start_candles_stream(self.asset, self.candle_duration, self.max_ticks)
        logging.info(f"Stream iniciado para {self.asset}")
        print(f"{green}Stream iniciado para {self.asset}")

        try:
            while True:
                current_second = datetime.datetime.now().second
                if current_second == self.tick_start_time:
                    print(f"{blue}{'=' * 30}")
                    print(f"{yellow}Iniciando recolección de ticks (Segundo {current_second}):")
                    self.ticks = []
                    candles = self.api.get_realtime_candles(self.asset, self.candle_duration)
                    start_time = time.time()
                    while time.time() - start_time < (self.check_time - self.tick_start_time):
                        candles = self.api.get_realtime_candles(self.asset, self.candle_duration)
                        latest_timestamp = max(candles.keys())
                        self.process_tick({'price': candles[latest_timestamp]['close']})
                        time.sleep(1)
                    action, digits_info, decision = self.analyze_pressure()
                    pred_result, pred_prob = None, None
                    if action:
                        pred_result, pred_prob = self.predict_result(digits_info, decision)
                        if pred_result:
                            print(f"{yellow}Predicción IA: {pred_result} con probabilidad de WIN: {pred_prob:.2f}")
                            if pred_result == 'WIN' and pred_prob >= self.win_threshold:
                                logging.info(f"Estrategia cumplida y aprobada por IA, ejecutando operación: {action}")
                                self.place_trade(action, digits_info, decision, pred_result, pred_prob)
                            else:
                                logging.info("IA desaconseja operar debido a baja probabilidad de WIN.")
                                print(f"{red}IA desaconseja operar.")
                                registrar_operacion(self.asset, action, 'NONE', 0, digits_info, decision, pred_result, pred_prob)
                        else:
                            logging.info(f"Estrategia cumplida, ejecutando operación sin IA: {action}")
                            self.place_trade(action, digits_info, decision, pred_result, pred_prob)
                    else:
                        logging.info("Estrategia no cumplida, no se realiza operación.")
                        registrar_operacion(self.asset, 'None', 'NONE', 0, digits_info, decision, pred_result, pred_prob)
                    time.sleep(self.candle_duration - self.check_time + 1)
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Bot detenido por el usuario.")
            print(f"{yellow}Bot detenido por el usuario.")
        finally:
            self.api.stop_candles_stream(self.asset, self.candle_duration)
            logging.info("Stream detenido.")
            print(f"{yellow}Stream detenido.")

def select_account_type():
    while True:
        print(f"{blue}\n{'-' * 50}")
        escolha = input(f"{yellow} >> {white}Selección de tipo de cuenta (Demo, Real o Torneo): {green}")
        print(f"{white}", end='')
        if escolha.lower() == 'demo':
            print(f"{yellow} >> {green}Cuenta Demo seleccionada{white}")
            return 'PRACTICE'
        elif escolha.lower() == 'real':
            print(f"{yellow} >> {green}Cuenta Real seleccionada{white}")
            return 'REAL'
        elif escolha.lower() == 'torneo':
            print(f"{yellow} >> {green}Cuenta de Torneo seleccionada{white}")
            return 'TOURNAMENT'
        else:
            print(f"{yellow} >> {red}Selección incorrecta! Digite 'Demo', 'Real' o 'Torneo'{white}")
        print(f"{blue}{'-' * 50}")

def main():
    print(f"{blue}{'-' * 50}")
    try:
        config = ConfigObj('config.txt')
        email = config['LOGIN']['email']
        password = config['LOGIN']['password']
        valor_entrada = config['AJUSTES']['valor_entrada']
    except Exception as e:
        print(f"{red}Error al leer el archivo de configuración: {e}")
        print(f"{yellow}Asegúrese de que el archivo 'config.txt' exista y tiene el formato correcto.")
        sys.exit()

    print(f"{yellow} >> {white}Configuración cargada:")
    print(f"{white}Email: {email}")
    print(f"{white}Valor de entrada: {valor_entrada}")
    print(f"{blue}{'-' * 50}")

    api = IQ_Option(email, password)
    if not api.connect():
        print(f"{red}Error: No se pudo conectar con la API de IQ Option.")
        sys.exit()

    while True:
        try:
            asset = input(f"{yellow} >> {white}Seleccione el activo (por ejemplo, USDZAR-OTC, EURJPY-OTC, etc.): {green}").strip()
            print(f"{white}", end='')
            account_type = select_account_type()
            bot = NumberPressureBot(email, password, valor_entrada, asset, account_type, api)
            print(f"{yellow}Activo seleccionado: {bot.asset}")
            bot.run()
            break
        except ValueError as e:
            print(f"{red}Error: {e}")
            print(f"{yellow}Por favor, ingrese un activo válido.")

if __name__ == "__main__":
    main()
