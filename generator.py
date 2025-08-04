import random
import csv

# Названия категорий и подкатегорий
categories = {
    "Окружение": ["Растительность", "Камни", "Препятствия", "Ландшафт"],
    "Противники": ["Внешний вид", "Атаки", "Анимации", "Звук"],
    "Босс": ["Внешний вид", "Атаки", "Анимации", "Звук"],
    "Лут": ["Тип", "Количество"],
    "Звуки": ["Окружение", "Саундтрек"]
}

# Специализация по подкатегориям
task_type_code_map = {
    "Растительность": 2, "Камни": 2, "Препятствия": 2, "Ландшафт": 3,
    "Внешний вид": 2, "Атаки": 1, "Анимации": 2, "Звук": 5,
    "Тип": 4, "Количество": 4,
    "Окружение": 5, "Саундтрек": 5
}

def generate_tasks():
    tasks_output = []
    task_id = 0

    num_locations = int(input("Введите количество локаций: "))
    
    for loc in range(1, num_locations + 1):
        num_levels = int(input(f"Введите количество уровней в локации {loc}: "))
        for lvl in range(1, num_levels + 1):
            for category, sub_items in categories.items():
                num_objects = int(input(f"Введите количество объектов для '{category}' на уровне {lvl} локации {loc}: "))
                for obj in range(1, num_objects + 1):
                    task_type = random.choice(sub_items)
                    name = f"Создать элемент '{task_type}' для локации {loc} уровня {lvl}, объект {obj}"
                    dif = random.randint(1, 5)
                    time = dif * random.randint(1, 3)  # более разнообразная генерация времени
                    priority = random.randint(1, 3)
                    min_exp = dif
                    code = task_type_code_map.get(task_type, 6)  # по умолчанию тестирование, если вдруг не задано
                    urgency = random.randint(1, 3)

                    task = {
                        "ID": task_id,
                        "name": name,
                        "dif": dif,
                        "time": time,
                        "priority": priority,
                        "min.exp": min_exp,
                        "code": code,
                        "urgency": urgency
                    }

                    tasks_output.append(task)
                    task_id += 1

    # Запись в CSV файл
    with open("generated_tasks.csv", "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["ID", "name", "dif", "time", "priority", "min.exp", "code", "urgency"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks_output:
            writer.writerow(task)

    print("Задачи успешно сгенерированы и сохранены в файл generated_tasks.csv")

# Запуск генератора
generate_tasks()
