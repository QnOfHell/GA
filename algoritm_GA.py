import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import timedelta

# Конфигурация
POP_SIZE = 75
GENERATIONS = 300
MUTATION_RATE = 0.15
ELITE_SIZE = 10
TOURNAMENT_SIZE = 25
MAX_WORKERS = 8

def load_data():
    tasks = np.genfromtxt('generated_tasks.csv', delimiter=',', skip_header=1, 
                         usecols=(0,1,2,3,4,5,6), dtype=np.float32)
    workers = np.genfromtxt('workers.csv', delimiter=',', skip_header=1,
                          usecols=(0,1,2,3,4,5), dtype=np.float32)
    return tasks, workers

tasks, workers = load_data()

# Предварительные вычисления
task_codes = tasks[:,6].astype(np.int32)
worker_codes = workers[:,4].astype(np.int32)
unique_codes = np.unique(task_codes)

valid_assignments = {}
for code in unique_codes:
    task_mask = task_codes == code
    worker_mask = worker_codes == code
    valid_assignments[code] = (task_mask, np.where(worker_mask)[0])

def calculate_duration(task, worker):
    base_time = task[3] / worker[5]
    if worker[2] >= task[5]:
        return base_time, 0  # Без штрафа
    return base_time * 1.5, base_time * 0.5  # С штрафом

def evaluate_individual(individual):
    worker_times = np.zeros(len(workers))
    penalty_time = 0.0
    for task_idx, worker_idx in enumerate(individual):
        if worker_idx == -1:
            continue
        duration, penalty = calculate_duration(tasks[task_idx], workers[worker_idx])
        worker_times[worker_idx] += duration
        penalty_time += penalty
    return worker_times, penalty_time

def evaluate_population(population):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(evaluate_individual, population))
    return results

def generate_population():
    population = []
    for _ in range(POP_SIZE):
        individual = np.empty(len(tasks), dtype=np.int32)
        for code in unique_codes:
            task_mask, code_workers = valid_assignments[code]
            n_tasks = task_mask.sum()
            assignments = np.tile(code_workers, (n_tasks // len(code_workers) + 1))[:n_tasks]
            np.random.shuffle(assignments)
            individual[task_mask] = assignments
        population.append(individual)
    return population

def tournament_selection(population, fitness):
    selected = []
    for _ in range(2):
        contenders = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
        best_idx = contenders[np.argmax(fitness[contenders])]
        selected.append(population[best_idx])
    return selected

def crossover(parent1, parent2):
    child1, child2 = parent1.copy(), parent2.copy()
    for code in unique_codes:
        if np.random.rand() < 0.7:
            task_mask = valid_assignments[code][0]
            crossover_point = np.random.randint(1, task_mask.sum())
            mask = np.where(task_mask)[0][:crossover_point]
            child1[mask], child2[mask] = parent2[mask], parent1[mask]
    return child1, child2

def mutate(individual):
    if np.random.rand() < MUTATION_RATE:
        code = np.random.choice(unique_codes)
        task_mask, code_workers = valid_assignments[code]
        if len(code_workers) > 1:
            worker_counts = np.bincount(individual[task_mask], minlength=len(workers))
            max_worker = np.argmax(worker_counts[code_workers])
            min_worker = np.argmin(worker_counts[code_workers])
            tasks_to_move = np.where((individual == code_workers[max_worker]) & task_mask)[0]
            if len(tasks_to_move) > 0:
                individual[np.random.choice(tasks_to_move)] = code_workers[min_worker]

def analyze_solution(solution):
    # Инициализация статистики
    stats = {
        'workers': defaultdict(lambda: {
            'tasks': 0,
            'time': 0.0,
            'penalty_tasks': 0,
            'penalty_time': 0.0
        }),
        'total': {
            'tasks': 0,
            'time': 0.0,
            'penalty_tasks': 0,
            'penalty_time': 0.0
        }
    }
    
    # Сбор данных
    for task_idx, worker_idx in enumerate(solution):
        if worker_idx == -1:
            continue
        
        task = tasks[task_idx]
        worker = workers[worker_idx]
        duration, penalty = calculate_duration(task, worker)
        
        stats['workers'][worker_idx]['tasks'] += 1
        stats['workers'][worker_idx]['time'] += duration
        stats['total']['tasks'] += 1
        stats['total']['time'] += duration
        
        if penalty > 0:
            stats['workers'][worker_idx]['penalty_tasks'] += 1
            stats['workers'][worker_idx]['penalty_time'] += penalty
            stats['total']['penalty_tasks'] += 1
            stats['total']['penalty_time'] += penalty
    
    # Вывод статистики
    print("\n=== Анализ распределения задач ===")
    print(f"Всего задач: {stats['total']['tasks']}")
    print(f"Штрафные задачи: {stats['total']['penalty_tasks']} "
          f"({stats['total']['penalty_tasks']/stats['total']['tasks']*100:.1f}%)")
    print(f"Доп. время из-за штрафов: {stats['total']['penalty_time']:.1f} ч "
          f"({stats['total']['penalty_time']/stats['total']['time']*100:.1f}% от общего времени)\n")
    
    print("● Загруженность исполнителей:")
    max_time = max(w['time'] for w in stats['workers'].values())
    for worker_idx, data in stats['workers'].items():
        status = "← MAX" if data['time'] == max_time else ""
        penalty_percent = data['penalty_tasks']/data['tasks']*100 if data['tasks'] > 0 else 0
        print(f"Исп. {worker_idx}: {data['tasks']} задач, {data['time']:.1f} ч")
    
    return stats

# НОВАЯ ФУНКЦИЯ: Визуализация диаграммы Ганта
def plot_gantt_chart(solution, tasks, workers, stats):
    # 1. Подготовка данных
    worker_tasks = defaultdict(list)
    worker_start_times = defaultdict(float)
    
    # Собираем задачи по исполнителям
    for task_idx, worker_idx in enumerate(solution):
        if worker_idx == -1:
            continue
        duration, _ = calculate_duration(tasks[task_idx], workers[worker_idx])
        worker_tasks[worker_idx].append({
            'task_idx': int(task_idx),
            'duration': duration,
            'start': worker_start_times[worker_idx],
            'end': worker_start_times[worker_idx] + duration,
            'type': int(tasks[task_idx, 6])  # Тип задачи из столбца 6
        })
        worker_start_times[worker_idx] += duration
    
    # 2. Создание диаграммы
    plt.figure(figsize=(14, 10))
    y_ticks = []
    y_labels = []
    
    # Цветовая карта для типов задач
    unique_task_types = np.unique(tasks[:,6].astype(int))
    color_map = plt.cm.get_cmap('tab20', len(unique_task_types))
    type_colors = {ttype: color_map(i) for i, ttype in enumerate(unique_task_types)}
    
    # Для каждого исполнителя
    for i, (worker_idx, tasks_list) in enumerate(worker_tasks.items()):
        y_pos = i
        y_ticks.append(y_pos)
        y_labels.append(f"Исп. {int(worker_idx)}\n({stats['workers'][worker_idx]['time']:.1f} ч)")
        
        # Для каждой задачи исполнителя
        for task in tasks_list:
            # Получаем цвет по типу задачи
            color = type_colors[task['type']]
            
            # Создаем горизонтальную полосу (задачу)
            plt.barh(y_pos, task['duration'], left=task['start'], 
                     color=color, edgecolor='black', alpha=0.8)
            
            # Добавляем текст (ID задачи)
            if task['duration'] > 1:  # Только для задач с достаточной длительностью
                plt.text(task['start'] + task['duration']/2, y_pos, 
                         f"T{task['task_idx']}", 
                         ha='center', va='center', color='black', fontsize=8)
    
    # 3. Настройка оформления
    plt.title('Диаграмма Ганта распределения задач (цвет по типу задачи)', fontsize=14)
    plt.xlabel('Время выполнения (часы)', fontsize=12)
    plt.ylabel('Исполнители', fontsize=12)
    plt.yticks(y_ticks, y_labels)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 4. Легенда для типов задач
    type_patches = [
        mpatches.Patch(color=color, label=f'Тип {ttype}') 
        for ttype, color in type_colors.items()
    ]
    plt.legend(handles=type_patches, title='Типы задач', loc='upper right', 
               bbox_to_anchor=(1.15, 1), fontsize=9)
    
    # 5. Информационные аннотации
    total_time = stats['total']['time']
    penalty_time = stats['total']['penalty_time']
    plt.figtext(0.5, 0.01, 
                f"Общее время проекта: {total_time:.1f} ч | Штрафное время: {penalty_time:.1f} ч ({penalty_time/total_time*100:.1f}%)", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 0.95, 1])  # Оставляем место для легенды
    plt.savefig('gantt_chart_by_type.png', dpi=300, bbox_inches='tight')
    plt.show()

def genetic_algorithm():
    population = generate_population()
    best_solution = None
    best_fitness = -np.inf
    
    for generation in range(GENERATIONS):
        # Оценка популяции
        results = evaluate_population(population)
        fitness = np.array([1/(np.max(times)+1e-6) for times, _ in results])
        
        # Обновление лучшего решения
        current_best = np.argmax(fitness)
        if fitness[current_best] > best_fitness:
            best_fitness = fitness[current_best]
            best_solution = population[current_best].copy()
        
        # Отбор и создание нового поколения
        new_population = [population[i] for i in np.argsort(fitness)[-ELITE_SIZE:]]
        while len(new_population) < POP_SIZE:
            parents = tournament_selection(population, fitness)
            child1, child2 = crossover(*parents)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population[:POP_SIZE]
        
        if generation % 10 == 0:
            print(f"Поколение {generation}: Лучшая приспособленность {best_fitness:.4f}")
    
    stats = analyze_solution(best_solution)
    # Визуализация диаграммы Ганта
    plot_gantt_chart(best_solution, tasks, workers, stats)
    return best_solution

if __name__ == "__main__":
    solution = genetic_algorithm()