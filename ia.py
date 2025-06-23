import pandas as pd
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def parse_historico(file_path='historico_operaciones.txt'):
    """Lee el archivo historico_operaciones.txt y extrae los datos relevantes."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Extraer dígitos y colores usando expresiones regulares
            digits_match = re.search(r'Análisis: \[(.*?)\]', line)
            result_match = re.search(r'Resultado: (\w+)', line)
            decision_match = re.search(r'Decisión: (.+?)(?:\n|$)', line)

            if digits_match and result_match:
                digits_str = digits_match.group(1)
                result = result_match.group(1)
                decision = decision_match.group(1) if decision_match else 'No se realiza operación'

                # Solo procesar operaciones con resultado WIN, LOSS o EMPATE
                if result in ['WIN', 'LOSS', 'EMPATE']:
                    # Extraer dígitos y colores
                    ticks = digits_str.split('; ')
                    digits = []
                    colors = []
                    for tick in ticks:
                        digit = re.search(r'Dígito=(\d+)', tick).group(1)
                        color = re.search(r'\((\w+)\)', tick).group(1)
                        digits.append(int(digit))
                        colors.append(color)

                    # Crear fila de datos
                    row = {
                        'digit1': digits[0], 'color1': colors[0],
                        'digit2': digits[1], 'color2': colors[1],
                        'digit3': digits[2], 'color3': colors[2],
                        'digit4': digits[3], 'color4': colors[3],
                        'decision': decision,
                        'result': result
                    }
                    data.append(row)

    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocesa los datos para el modelo."""
    # Codificar colores y decisión
    le_color = LabelEncoder()
    le_decision = LabelEncoder()
    le_result = LabelEncoder()

    df['color1'] = le_color.fit_transform(df['color1'])
    df['color2'] = le_color.transform(df['color2'])
    df['color3'] = le_color.transform(df['color3'])
    df['color4'] = le_color.transform(df['color4'])
    df['decision'] = le_decision.fit_transform(df['decision'])
    df['result'] = le_result.fit_transform(df['result'])

    # Guardar los codificadores para usarlos en predicciones futuras
    joblib.dump(le_color, 'le_color.pkl')
    joblib.dump(le_decision, 'le_decision.pkl')
    joblib.dump(le_result, 'le_result.pkl')

    return df, le_result

def train_model(df, le_result):
    """Entrena un modelo Random Forest con validación cruzada y optimización de hiperparámetros."""
    X = df[['digit1', 'color1', 'digit2', 'color2', 'digit3', 'color3', 'digit4', 'color4', 'decision']]
    y = df['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir el modelo base
    model = RandomForestClassifier(random_state=42)

    # Realizar validación cruzada
    scores = cross_val_score(model, X, y, cv=5)
    logging.info(f'Precisión promedio (validación cruzada): {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')

    # Optimizar hiperparámetros con GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f'Mejores parámetros: {grid_search.best_params_}')

    # Obtener el mejor modelo
    model = grid_search.best_estimator_

    # Evaluar el modelo en el conjunto de prueba
    y_pred = model.predict(X_test)
    logging.info(f'Reporte de clasificación:\n{classification_report(y_test, y_pred, target_names=le_result.classes_)}')

    # Guardar el modelo
    joblib.dump(model, 'digit_classifier.pkl')

    return model, le_result

def main():
    logging.info("Iniciando entrenamiento del modelo...")
    df = parse_historico()
    if df.empty:
        logging.error("No se encontraron datos válidos en historico_operaciones.txt")
        return

    logging.info(f"Datos cargados: {len(df)} operaciones")
    df, le_result = preprocess_data(df)
    model, le_result = train_model(df, le_result)
    logging.info("Modelo entrenado y guardado como 'digit_classifier.pkl'")

if __name__ == "__main__":
    main()
