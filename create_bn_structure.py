import json
import os

# Имя файла конфигурации (должен лежать в той же папке, что и скрипт)
CONFIG_FILENAME = 'bn_config.json'
# Имя файла для сохранения результатов по умолчанию
DEFAULT_OUTPUT_FILENAME = 'incident_evidence.json'

def load_config(filepath=CONFIG_FILENAME):
    """Загружает конфигурацию факторов и состояний из JSON-файла."""
    if not os.path.exists(filepath):
        print(f"Ошибка: Файл конфигурации '{filepath}' не найден.")
        print("Убедитесь, что файл bn_config.json находится в той же папке, что и скрипт.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        # Простая валидация структуры
        if 'factors' not in config_data or not isinstance(config_data['factors'], list):
             raise ValueError("Некорректный формат: отсутствует ключ 'factors' или он не является списком.")
        if 'outcome_node' not in config_data:
            print("Предупреждение: в конфигурации отсутствует ключ 'outcome_node'.")
        if 'category_mapping' not in config_data:
             print("Предупреждение: в конфигурации отсутствует ключ 'category_mapping'.")

        print(f"Файл конфигурации '{filepath}' успешно загружен.")
        return config_data
    except json.JSONDecodeError as e:
        print(f"Ошибка: Не удалось декодировать JSON из файла '{filepath}'.")
        print(f"Детали ошибки: {e}")
        return None
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при загрузке конфигурации: {e}")
        return None

def get_user_input_for_factor(factor_data):
    """
    Запрашивает у пользователя состояние для одного фактора.

    Args:
        factor_data (dict): Словарь с информацией о факторе из конфиг. файла.

    Returns:
        int or None: Числовой код выбранного состояния или None, если пропущено.
    """
    print("-" * 40)
    print(f"Фактор: {factor_data['name']} (ID: {factor_data['id']})")
    print("Возможные состояния:")

    valid_choices = {}
    for i, state in enumerate(factor_data['states']):
        choice_num = i + 1
        print(f"  {choice_num}: {state['label']}")
        valid_choices[str(choice_num)] = state['code'] # Сохраняем код состояния

    print("  (Нажмите Enter, чтобы пропустить этот фактор)")

    while True:
        user_choice = input("Ваш выбор: ").strip()

        if not user_choice: # Пользователь нажал Enter (пропустить)
            print("Фактор пропущен.")
            return None
        elif user_choice in valid_choices:
            selected_code = valid_choices[user_choice]
            # Найдем label для подтверждения
            selected_label = ""
            for state in factor_data['states']:
                if state['code'] == selected_code:
                    selected_label = state['label']
                    break
            print(f"Выбрано: '{selected_label}' (код: {selected_code})")
            return selected_code
        else:
            print("Некорректный ввод. Пожалуйста, введите номер состояния из списка или нажмите Enter.")

def collect_evidence_from_user(config_data):
    """
    Собирает данные (свидетельства) от пользователя для всех факторов.

    Args:
        config_data (dict): Загруженная конфигурация.

    Returns:
        dict: Словарь со свидетельствами {factor_id: state_code}.
              Включает только те факторы, для которых пользователь указал состояние.
    """
    evidence = {}
    if not config_data or 'factors' not in config_data:
        print("Ошибка: Невозможно собрать данные без корректной конфигурации.")
        return evidence

    print("\n--- Сбор данных об авиационном происшествии ---")
    print("Для каждого фактора выберите соответствующее состояние или нажмите Enter, чтобы пропустить.")

    factors = config_data['factors']
    total_factors = len(factors)

    for i, factor in enumerate(factors):
        print(f"\nФактор {i+1} из {total_factors}")
        factor_id = factor.get('id')
        if not factor_id:
            print(f"Предупреждение: Пропуск фактора без 'id' в конфигурации: {factor.get('name', 'Неизвестное имя')}")
            continue

        # Проверка наличия состояний
        if not factor.get('states') or not isinstance(factor['states'], list):
             print(f"Предупреждение: Пропуск фактора '{factor.get('name')}' из-за отсутствия или некорректного формата 'states'.")
             continue

        selected_code = get_user_input_for_factor(factor)
        if selected_code is not None: # Только если пользователь не пропустил
            evidence[factor_id] = selected_code

    return evidence

def save_evidence_to_json(evidence, output_filename):
    """
    Сохраняет собранные свидетельства в JSON-файл.

    Args:
        evidence (dict): Словарь со свидетельствами.
        output_filename (str): Имя файла для сохранения.
    """
    try:
        # Добавляем обертку, чтобы JSON был структурирован
        output_data = {"evidence": evidence}
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"\nДанные успешно сохранены в файл: '{output_filename}'")
    except IOError as e:
        print(f"Ошибка: Не удалось записать данные в файл '{output_filename}'.")
        print(f"Детали ошибки: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при сохранении файла: {e}")

# --- Основная часть программы ---
if __name__ == "__main__":
    # 1. Загрузка конфигурации
    configuration = load_config()

    if configuration:
        # 2. Сбор данных от пользователя
        collected_evidence = collect_evidence_from_user(configuration)

        # 3. Запрос имени выходного файла
        output_filename_input = input(f"\nВведите имя файла для сохранения данных \n(нажмите Enter для '{DEFAULT_OUTPUT_FILENAME}'): ").strip()
        output_filename = output_filename_input if output_filename_input else DEFAULT_OUTPUT_FILENAME

        # Убедимся, что у файла расширение .json (необязательно, но хорошая практика)
        if not output_filename.lower().endswith('.json'):
            output_filename += '.json'

        # 4. Сохранение результатов
        save_evidence_to_json(collected_evidence, output_filename)

    else:
        print("\nПрограмма не может продолжить работу без файла конфигурации.")

    # Ожидание ввода пользователя перед закрытием консоли (удобно при запуске двойным кликом)
    input("\nНажмите Enter для завершения...")