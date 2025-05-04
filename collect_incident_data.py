import json
import os
#from pgmpy.models import BayesianNetwork
from pgmpy.models import DiscreteBayesianNetwork
# NetworkXError может понадобиться для отлова ошибки циклов
from networkx.exception import NetworkXError

# --- 1. Загрузка Конфигурации и Определение Узлов ---
CONFIG_FILENAME = 'bn_config.json'

def load_nodes_from_config(filepath=CONFIG_FILENAME):
    """Загружает список ID узлов из конфигурационного JSON-файла."""
    if not os.path.exists(filepath):
        print(f"Ошибка: Файл конфигурации '{filepath}' не найден.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Извлекаем ID из списка факторов
        node_ids = {factor['id'] for factor in config_data.get('factors', []) if 'id' in factor}

        # Добавляем узлы категорий, если они есть и еще не добавлены
        category_nodes = {node_id for node_id in config_data.get('category_mapping', {}).values() if node_id}
        node_ids.update(category_nodes)

        # Добавляем узел исхода, если он есть и еще не добавлен
        outcome_node = config_data.get('outcome_node')
        if outcome_node:
            node_ids.add(outcome_node)

        if not node_ids:
             print("Ошибка: Не найдено ни одного ID узла в конфигурационном файле.")
             return None

        print(f"Успешно загружено {len(node_ids)} уникальных ID узлов из '{filepath}'.")
        return list(node_ids) # Возвращаем список

    except json.JSONDecodeError as e:
        print(f"Ошибка: Не удалось декодировать JSON из файла '{filepath}'. Проверьте синтаксис JSON.")
        print(f"Детали ошибки: {e}")
        return None
    except KeyError as e:
        print(f"Ошибка: В файле конфигурации отсутствует ожидаемый ключ: {e}")
        return None
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при загрузке узлов из конфигурации: {e}")
        return None

# --- 2. Определение Структуры Рёбер ---
# !!! ВАЖНО: Вставьте сюда ПОЛНЫЙ список bn_structure_edges из предыдущего ответа !!!
bn_structure_edges = [
    # === Связи, идущие от Организационных факторов (Org) ===
    # Гос. надзор влияет на ...
    ("Org_RegulatoryOversight", "Org_SafetyCulture"),
    ("Org_RegulatoryOversight", "Org_MaintOrganization"),
    ("Org_RegulatoryOversight", "Org_Oversight"),
    # Культура безопасности влияет на ...
    ("Org_SafetyCulture", "Org_ProceduresQuality"),
    ("Org_SafetyCulture", "Org_TrainingQuality"),
    ("Org_SafetyCulture", "Org_MaintOrganization"),
    ("Org_SafetyCulture", "Org_Oversight"),
    ("Org_SafetyCulture", "HF_Violation"),
    ("Org_SafetyCulture", "Org_Pressure"),
    # Качество процедур влияет на ...
    ("Org_ProceduresQuality", "HF_SkillError"),
    ("Org_ProceduresQuality", "HF_DecisionError"),
    ("Org_ProceduresQuality", "HF_Violation"),
    # Качество подготовки влияет на ...
    ("Org_TrainingQuality", "HF_KnowledgeSkill"),
    ("Org_TrainingQuality", "HF_CRM"),
    # Качество организации ТО влияет на ...
    ("Org_MaintOrganization", "HF_Maint_Quality"),
    ("Org_MaintOrganization", "Tech_SystemFailure"),
    # ("Org_MaintOrganization", "Tech_ManufacturingDefect"), # Спорно
    # Внутренний контроль АК влияет на ...
    ("Org_Oversight", "HF_Violation"),
    ("Org_Oversight", "HF_Maint_Quality"),
    # Ресурсы влияют на ...
    ("Org_Resources", "Org_PlanningScheduling"),
    ("Org_Resources", "Org_MaintOrganization"),
    ("Org_Resources", "Org_TrainingQuality"),
    # Планирование/Расписание влияет на ...
    ("Org_PlanningScheduling", "HF_Fatigue"),
    # Производственное давление влияет на ...
    ("Org_Pressure", "HF_Violation"),
    ("Org_Pressure", "HF_Maint_Pressure"),
    ("Org_Pressure", "HF_DecisionError"),

    # === Связи, идущие от Факторов среды (Env) ===
    ("Env_Icing", "Ctx_TaskComplexity"),
    ("Env_Icing", "HF_PerceptionError"),
    ("Env_Icing", "Tech_SystemFailure"),
    ("Env_Turbulence", "Ctx_TaskComplexity"),
    ("Env_Turbulence", "HF_SkillError"),
    ("Env_WindShear", "Ctx_TaskComplexity"),
    ("Env_WindShear", "HF_SkillError"),
    ("Env_Thunderstorm", "Ctx_TaskComplexity"),
    ("Env_Thunderstorm", "Env_Turbulence"),
    ("Env_Thunderstorm", "Env_Icing"),
    ("Env_Thunderstorm", "Env_WindShear"),
    # ("Env_Thunderstorm", "Ext_UnlawfulInterference"), # Молния - не причина грозы, а следствие
    ("Env_Visibility", "Ctx_TaskComplexity"),
    ("Env_Visibility", "HF_PerceptionError"),
    ("Env_Ceiling", "Ctx_TaskComplexity"),
    ("Env_Ceiling", "HF_PerceptionError"),
    ("Env_SurfaceWind", "Ctx_TaskComplexity"),
    ("Env_SurfaceWind", "HF_SkillError"),
    ("Env_Temperature", "Tech_SystemFailure"),
    ("Env_VolcanicAsh", "Tech_SystemFailure"),
    ("Env_VolcanicAsh", "Ctx_TaskComplexity"),
    ("Env_RunwayCondition", "Ctx_TaskComplexity"),
    ("Env_RunwayCondition", "HF_SkillError"),
    ("Env_AirportLights", "Ctx_TaskComplexity"),
    ("Env_AirportLights", "HF_PerceptionError"),
    ("Env_NavAids", "Ctx_TaskComplexity"),
    ("Env_NavAids", "HF_DecisionError"),
    ("Env_Obstacles", "Ctx_TaskComplexity"),
    ("Env_Terrain", "Ctx_TaskComplexity"),
    ("Env_TrafficDensity", "Ctx_Workload"),
    ("Env_TrafficDensity", "HF_ATC_Workload"),
    ("Env_AirspaceComplexity", "Ctx_Workload"),
    ("Env_AirspaceComplexity", "HF_SkillError"),
    ("Env_BirdstrikeRisk", "Tech_SystemFailure"),

    # === Связи, идущие от Технических факторов (Tech) ===
    ("Tech_DesignFlaw", "Tech_SystemFailure"),
    ("Tech_ManufacturingDefect", "Tech_SystemFailure"),
    ("Tech_SystemFailure", "HF_DecisionError"),
    ("Tech_SystemFailure", "HF_SkillError"),
    ("Tech_SystemFailure", "Ctx_TaskComplexity"),
    ("Tech_Fire", "Ctx_TaskComplexity"),
    ("Tech_Fire", "HF_DecisionError"),
    ("Tech_Fire", "Outcome_Severity"),

    # === Связи, идущие от Человеческого фактора (HF) - предпосылки ===
    ("HF_Fatigue", "HF_SkillError"),
    ("HF_Fatigue", "HF_DecisionError"),
    ("HF_Fatigue", "HF_PerceptionError"),
    ("HF_Stress", "HF_SkillError"),
    ("HF_Stress", "HF_DecisionError"),
    ("HF_Medical", "HF_SkillError"),
    ("HF_Medical", "HF_DecisionError"),
    ("HF_Complacency", "HF_DecisionError"),
    ("HF_Complacency", "HF_Violation"),
    ("HF_Distraction", "HF_SkillError"),
    ("HF_Distraction", "HF_PerceptionError"),
    ("HF_KnowledgeSkill", "HF_SkillError"),
    ("HF_KnowledgeSkill", "HF_DecisionError"),
    ("HF_Experience", "HF_SkillError"),
    ("HF_Experience", "HF_DecisionError"),
    ("HF_Motivation", "HF_Violation"),
    ("HF_CRM", "HF_SkillError"),
    ("HF_CRM", "HF_DecisionError"),
    ("HF_ATC_CommError", "HF_PerceptionError"),
    ("HF_ATC_InstructionError", "HF_DecisionError"),
    ("HF_ATC_Separation", "Outcome_Severity"),
    ("HF_ATC_InfoLack", "HF_DecisionError"),
    ("HF_ATC_Workload", "HF_ATC_CommError"),
    ("HF_ATC_Workload", "HF_ATC_InstructionError"),
    ("HF_Maint_Quality", "Tech_SystemFailure"),
    ("HF_Maint_Pressure", "HF_Maint_Quality"),
    ("HF_Ground_FuelError", "Tech_SystemFailure"),
    ("HF_Ground_LoadError", "Tech_SystemFailure"), # Влияет на управляемость -> может привести к отказу? Или к HF_SkillError?
    ("HF_Ground_Damage", "Tech_SystemFailure"),
    # ("HF_Ground_Deicing", "Env_Icing"), # Не ПОО вызывает обледенение, а предотвращает. Связь сложнее.
    ("HF_Ground_Deicing", "Tech_SystemFailure"), # Некачественная ПОО -> отказ системы из-за льда
    ("HF_Ground_FOD", "Tech_SystemFailure"),

    # === Связи, идущие от Контекста полета (Ctx) ===
    ("Ctx_TaskComplexity", "HF_SkillError"),
    ("Ctx_TaskComplexity", "HF_DecisionError"),
    ("Ctx_Workload", "HF_SkillError"),
    ("Ctx_Workload", "HF_DecisionError"),
    ("Ctx_Workload", "HF_Fatigue"),
    ("Ctx_UnexpectedEvents", "Ctx_TaskComplexity"),
    ("Ctx_UnexpectedEvents", "Ctx_Workload"),
    ("Ctx_UnexpectedEvents", "HF_DecisionError"),
    ("Ctx_FlightPhase", "Ctx_TaskComplexity"),
    ("Ctx_FlightPhase", "Ctx_Workload"),
    ("Ctx_FlightPhase", "HF_SkillError"),

    # === Связи, ведущие к Исходу (Outcome) ===
    ("HF_SkillError", "Outcome_Severity"),
    ("HF_DecisionError", "Outcome_Severity"),
    ("HF_PerceptionError", "Outcome_Severity"),
    ("HF_Violation", "Outcome_Severity"),
    ("Tech_SystemFailure", "Outcome_Severity"),
    ("Ext_UnlawfulInterference", "Outcome_Severity"),
    ("Env_WindShear", "Outcome_Severity"), # Сильный сдвиг может быть прямой причиной
    ("Env_Visibility", "Outcome_Severity"), # CFIT
    ("Env_Terrain", "Outcome_Severity"), # CFIT
    # ("Env_BirdstrikeRisk", "Outcome_Severity"), # Обычно через Tech_SystemFailure

    # === Связи к узлам-категориям (если они есть) ===
    # Эти связи нужны, если узлы типа 'ВнешнУсл' есть в модели
    # Если их нет, этот блок и соответствующий маппинг в JSON не нужны для структуры
    # Пример для нескольких:
    ("Env_Icing", "ВнешнУсл"),
    ("Env_Visibility", "ВнешнУсл"),
    ("HF_SkillError", "ЧелФактор"),
    ("HF_Fatigue", "ЧелФактор"),
    ("Tech_SystemFailure", "ТехничФактор"),
    ("Org_SafetyCulture", "ОрганизФактор"),

    # === Связи МЕЖДУ узлами категорий (если они есть) ===
    ("ОрганизФактор", "ЧелФактор"),
    ("ОрганизФактор", "ТехничФактор"),
    ("ВнешнУсл", "ЧелФактор"),
    ("ВнешнУсл", "ТехничФактор"),
    ("ТехничФактор", "ЧелФактор"),
    # Связи от категорий к исходу
    ("ЧелФактор", "Outcome_Severity"),
    ("ТехничФактор", "Outcome_Severity"),
    ("ВнешнУсл", "Outcome_Severity"),
    ("ОрганизФактор", "Outcome_Severity") # Организационные проблемы редко прямая причина, но могут быть
]
# Убедитесь, что вы вставили ВЕСЬ список сюда

# --- Основная часть программы ---
if __name__ == "__main__":
    print("Запуск скрипта создания и проверки структуры БС...")

    # 1. Загрузка узлов
    print(f"\n--- Шаг 1: Загрузка узлов из '{CONFIG_FILENAME}' ---")
    node_ids = load_nodes_from_config()

    if node_ids:
        # 2. Создание модели
        print("\n--- Шаг 2: Создание объекта BayesianNetwork ---")
        try:
            model = DiscreteBayesianNetwork()
            print("Объект модели успешно создан.")
        except Exception as e:
            print(f"Ошибка при создании объекта модели: {e}")
            exit() # Выход, если даже объект создать не удалось

        # 3. Добавление узлов
        print("\n--- Шаг 3: Добавление узлов в модель ---")
        try:
            model.add_nodes_from(node_ids)
            print(f"Успешно добавлено {len(model.nodes())} узлов в модель.")
            # Проверка, все ли узлы из списка добавлены
            if len(model.nodes()) != len(node_ids):
                 print(f"Предупреждение: Ожидалось {len(node_ids)} узлов, но добавлено {len(model.nodes())}. Возможны дубликаты ID в конфиге?")
        except Exception as e:
            print(f"Ошибка при добавлении узлов в модель: {e}")
            exit()

        # 4. Добавление рёбер
        print("\n--- Шаг 4: Добавление рёбер в модель ---")
        # Проверка, что все узлы, упомянутые в ребрах, существуют в модели
        edge_nodes = set(node for edge in bn_structure_edges for node in edge)
        missing_nodes_in_edges = edge_nodes - set(model.nodes())

        if missing_nodes_in_edges:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Следующие узлы, указанные в рёбрах, отсутствуют в списке узлов модели:")
            print(missing_nodes_in_edges)
            print("Невозможно добавить рёбра. Проверьте список рёбер 'bn_structure_edges' и конфигурацию узлов.")
            exit()
        else:
            try:
                model.add_edges_from(bn_structure_edges)
                print(f"Успешно добавлено {len(model.edges())} рёбер в модель.")
            except Exception as e:
                print(f"Ошибка при добавлении рёбер в модель: {e}")
                exit()

        # 5. Проверка модели на корректность (включая ацикличность)
        print("\n--- Шаг 5: Проверка модели на корректность (ацикличность и т.д.) ---")
        try:
            # Метод check_model() возвращает True если всё хорошо, иначе вызывает исключение
            model.check_model()
            print("УСПЕХ: Модель успешно прошла проверку. Структура является корректным DAG.")

        except NetworkXError as e:
            print(f"ОШИБКА ВАЛИДАЦИИ: Модель содержит циклы!")
            print(f"Детали ошибки NetworkX: {e}")
            # Попытка найти циклы (может быть ресурсоемко для больших графов)
            try:
                from networkx import simple_cycles
                cycles = list(simple_cycles(model))
                if cycles:
                    print("Найденные циклы (может быть не полный список для больших графов):")
                    # Выводим несколько первых циклов для примера
                    for i, cycle in enumerate(cycles[:5]):
                        print(f"  Цикл {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
                    if len(cycles) > 5:
                        print(f"  ... (всего найдено {len(cycles)} циклов)")
                else:
                    print("Не удалось точно определить циклы, но ошибка NetworkX указывает на их наличие.")
            except Exception as cycle_e:
                print(f"Не удалось найти конкретные циклы: {cycle_e}")

        except Exception as e:
            print(f"ОШИБКА ВАЛИДАЦИИ: Модель не прошла проверку.")
            print(f"Детали ошибки: {e}")
            print("Возможные причины: несвязный граф, некорректные узлы/рёбра (хотя это должно было проявиться раньше).")

    else:
        print("\nНе удалось загрузить узлы. Создание и проверка модели невозможны.")

    print("\nСкрипт завершил работу.")