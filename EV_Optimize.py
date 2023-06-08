# импорт необходимых библиотек
import random
import argparse
import dimod
import sys
import networkx as nx
import numpy as np
from dwave.system import LeapHybridSampler

import matplotlib

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt


# ===Функция чтения параметров модели (или использование установленных по умолчанию)
def input_params():
    # установка параметров пользователя (необязательная процедура)
    user_p = argparse.ArgumentParser()
    user_p.add_argument("-s", "--seed", help="установка случайного начального числа для сценария", type=int)
    user_p.add_argument("-x", "--width", help="установка ширины сетки", default=15, type=int)
    user_p.add_argument("-y", "--height", help="установка высоты сетки", default=15, type=int)
    user_p.add_argument("-p", "--poi", help="установка количества POIs", default=3, type=int)
    user_p.add_argument("-c", "--chargers", help="установка количества существующих станций", default=4, type=int)
    user_p.add_argument("-n", "--new-chargers", help="установка количества новыз станций", default=2, type=int)
    params = user_p.parse_args()

    # чтение заданных пользователем значений
    width_current = params.width  # шмрмеа сетки
    height_current = params.height  # высота сетки
    poiQuantity_current = params.poi  # кол-во poi (точек интереса)
    csQuantity_current = params.chargers  # кол-во установленных станций запядки
    csNewQuantity_current = params.new_chargers  # кол-во новых станций зарядки

    # установка начального числа (если оно указано пользователем)
    if (params.seed):
        random.seed(params.seed)

    # проверка введенных значений на неотрицательность
    if (width_current < 0) or (height_current < 0) or (poiQuantity_current < 0) or (csQuantity_current < 0) or (
            csNewQuantity_current < 0):
        print("Все значения параметров должны быть неотрицательными!")
        sys.exit(0)  # выход из приложения

    # проверка размеров сетки на возможность оптимизации
    if (poiQuantity_current > width_current * height_current) or (
            csQuantity_current + csNewQuantity_current > width_current * height_current):
        print("Размер сетки недостаточен для проведения оптимизации.")
        sys.exit(0)  # выход из приложения

    return params


# ===Функция настройки модели оптимизации
def model_settings(width_cur, height_cur, poiQuantity_cur, csQuantity_cur):
    """
    настройка модели с заданными параметрами

    Параметры:
        width_cur (int): ширина сетки
        height_cur (int): высота сетки
        poiQuantity_cur (int): кол-во точек интереса
        csQuantity_cur (int): кол-во существующих станций зарядки

    Возвращает:
        Net_Graph (граф сети): граф модели размером width_cur на height_cur
        pois_place (список кортежей целых чисел): фиксированный набор точек интереса
        cs_place (список кортежей целых чисел):
            набор расположений текущих станций зарядки
        csNewPotential_place (сеисок кортежей целых чисел):
            возможные новые места расположения зарядных станций
    """

    Net_Graph = nx.grid_2d_graph(width_cur, height_cur)
    points = list(Net_Graph.nodes)

    # определение нахождения фиксированного набора точек интереса
    pois_place = random.sample(points, k=poiQuantity_cur)

    # определение текущих мест расположения зарядных станций
    cs_place = random.sample(points, k=csQuantity_cur)

    # определение потенциальных новых мест зарядки
    csNewPotential_place = list(Net_Graph.nodes() - cs_place)

    return Net_Graph, pois_place, cs_place, csNewPotential_place


# ===Функция вычисления расстояния между двумя числами
def calc_length(param_1, param_2):
    return (param_1[0] ** 2 - 2 * param_1[0] * param_2[0] + param_2[0] ** 2) + (
                param_1[1] ** 2 - 2 * param_1[1] * param_2[1] + param_2[1] ** 2)


# === Формирование модели для решения
def form_modelBQM(csNewPotential_place, poiQuantity, pois_place, csQuantity, cs_place, csNewQuantity):
    """Формирование bqm для моделирования задачи для гибридного решателя

    Параметры:
        csNewPotential_place (список кортежей целых чисел):
            возможные новые места расположения станций зарядки
        poiQuantity (int): кол-во точек интереса
        pois_place (список кортежей целых чисел): фиксированный набор точек интереса
        csQuantity (int): кол-во существующих зарядных станций
        cs_place (фиксированный набор расположений точек интереса):
            набор расположений текущих зарядных станций
        csNewQuantity (int): желаемое кол-во новых зарядных станций

    ВОзвращает:
        bqm_np (BinaryQuadraticModel): модель QUBO для входного сценария
    """

    # настраиваемые параметры
    gam1_coef = len(csNewPotential_place) * 4
    gam_2_coef = len(csNewPotential_place) / 3
    gam_3_coef = len(csNewPotential_place) * 1.7
    gam_4_coef = len(csNewPotential_place) ** 3

    # формирование BQM с использованием adjVectors для нахождения оптимальных мест для старций т.е. min
    # расстояний до POIs и max расстояния к существующим станциям
    bqm_res = dimod.BinaryQuadraticModel(len(csNewPotential_place), 'BINARY')

    # Условие 1: Min среднее расстояние до POIs
    if poiQuantity > 0:
        for i in range(len(csNewPotential_place)):
            # вычисление расстояния дo POIs из этого узла
            current_distance = csNewPotential_place[i]
            average_distance = sum(calc_length(current_distance, cur) for cur in pois_place) / poiQuantity
            bqm_res.linear[i] += average_distance * gam1_coef

    # Условие 2: Max расстояние до существующих станций зарядки
    if csQuantity > 0:
        for i in range(len(csNewPotential_place)):
            # вычисление расстояния дo POIs из этого узла
            current_distance = csNewPotential_place[i]
            average_distance = -sum(calc_length(current_distance, cur)
                                    for cur in cs_place) / csQuantity
            bqm_res.linear[i] += average_distance * gam_2_coef

    # Условие 3: Max расстояние между другими станциями зарядки
    if csNewQuantity > 1:
        for i in range(len(csNewPotential_place)):
            for j in range(i + 1, len(csNewPotential_place)):
                param_i = csNewPotential_place[i]
                param_j = csNewPotential_place[j]
                temp_distance = -calc_length(param_i, param_j)
                bqm_res.add_interaction(i, j, temp_distance * gam_3_coef)

    # Условие 4: выбор точного кол-ва расположений csNewPotential_place
    bqm_res.update(dimod.generators.combinations(bqm_res.variables, csNewQuantity, strength=gam_4_coef))

    return bqm_res


# ===Функция запуска решателя для определения мест расположения зарядных станций
def goBQM(bqm, s_loc, csNewPotential_place, **kwargs):
    """Решение BQM с помощью семплера

    Параметры:
        bqm (BinaryQuadraticModel): модель QUBO для задачи
        s_loc: семплер для задачи
        csNewPotential_place (список кортежей целых чисе):
            возможніе новіе места расположения зарядніх станций
        **kwargs: Sampler-specific parameters to be used

    Возвоащает:
        csNew_place (список кортежей целых чисе):
            расположения новых зарядных станций
    """

    s_set = s_loc.sample(bqm,
                         label='EV Charger Stations Locations',
                         **kwargs)

    res = s_set.first.sample
    csNew_place = [csNewPotential_place[k] for k, v in res.items() if v == 1]

    return csNew_place


# ===Функция вывода решения на экран
def printSolve(pois_place, poiQuantity, cs_place, csQuantity, csNew_place, csNewQuantity):
    """Вывод статистики решения

    Параметры:
        pois_place (список кортежей целых чисел): фиксированный набор точек интереса
        poiQuantity (int): кол-во точек интереса
        cs_place (список кортежей целых чисел):
            фиксированный набор расположений зарядных станций
        csQuantity (int): кол-во существующих зарядных станций
        csNew_place (список кортежей целых чисел):
            расположения новых зарядных станций
        csNewQuantity (int): кол-во новых зарядных станций

    ВОзвращает:
        ничего.
    """

    print("\nПолученное решение: \n=====================")

    print("\nРазмещение новых зарядных станций:\t\t\t\t", csNew_place)

    if poiQuantity > 0:
        poi_average_distance = [0] * len(csNew_place)
        for poi_cur in pois_place:
            for i, cs_new in enumerate(csNew_place):
                poi_average_distance[i] += sum(
                    abs(param_1 - param_2) for param_1, param_2 in zip(cs_new, poi_cur)) / poiQuantity
        print("Среднее расстояние до POI:\t\t\t", poi_average_distance)

    if csQuantity > 0:
        old_cs_average_distance = [
            sum(abs(param_1 - param_2) for param_1, param_2 in zip(new_cs, cs_cur) for cs_cur in cs_place) / csQuantity
            for new_cs in csNew_place]
        print("Среднее расстояние до существующих зарядных станций:\t", old_cs_average_distance)

    if csNewQuantity > 1:
        new_cs_distance = 0
        for i in range(csNewQuantity):
            for j in range(i + 1, csNewQuantity):
                new_cs_distance += abs(csNew_place[i][0] - csNew_place[j][0]) + abs(
                    csNew_place[i][1] - csNew_place[j][1])
        print("Расстояние между новыми зарядными станциями:\t\t\t", new_cs_distance)


# ===Функция сохранения результатов оптимизации в графический файл
def saveResults(Net_Graph, pois_place, cs_place, csNew_place):
    """ Создание файла изображения с решенной моделью.
            - черные точки: доступный узел для размещения
            - красные точки: существующая станция зарядки
            - узлы с пометкой 'P': расположение POI
            - синие узлы: новые станции зарядки

    Параметры:
        Net_Graph (сетевой граф): граф сетки размером width x height
        pois_place (список кортежей целых чисел): расположения POI
        cs_place (список кортежей целых чисел):
            Расположения станций зарядки
        csnew_place (список кортежей целых чисел):
            Расположение новых станций зарядки

    Возвращает:
        Ничего. Результаты моделирования сохранены в "result.png".
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Расположение новых станций зарядки')
    pos = {x: [x[0], x[1]] for x in Net_Graph.nodes()}

    # Размещение POIs на карте
    poi_graph = Net_Graph.subgraph(pois_place)
    poi_labels = {x: 'P' for x in poi_graph.nodes()}

    # Размещение на карте старых зарядных станций
    cs_graph = Net_Graph.subgraph(cs_place)

    # Размещение на карте POIs и старых зарядных станций
    poi_cs_list = set(pois_place) - (set(pois_place) - set(cs_place))
    poi_cs_graph = Net_Graph.subgraph(poi_cs_list)
    poi_cs_labels = {x: 'P' for x in poi_graph.nodes()}

    # Отрисовка старой карты (левая часть изображения)
    nx.draw_networkx(Net_Graph, ax=ax1, pos=pos, with_labels=False, node_color='k', font_color='w')
    nx.draw_networkx(poi_graph, ax=ax1, pos=pos, with_labels=True,
                     labels=poi_labels, node_color='k', font_color='w')
    nx.draw_networkx(cs_graph, ax=ax1, pos=pos, with_labels=False, node_color='r',
                     font_color='k')
    nx.draw_networkx(poi_cs_graph, ax=ax1, pos=pos, with_labels=True,
                     labels=poi_cs_labels, node_color='r', font_color='w')

    # Отрисока новой карты (правая часть изображения)
    new_cs_graph = Net_Graph.subgraph(csNew_place)
    nx.draw_networkx(Net_Graph, ax=ax2, pos=pos, with_labels=False, node_color='k',
                     font_color='w')
    nx.draw_networkx(poi_graph, ax=ax2, pos=pos, with_labels=True,
                     labels=poi_labels, node_color='k', font_color='w')
    nx.draw_networkx(cs_graph, ax=ax2, pos=pos, with_labels=False, node_color='r',
                     font_color='k')
    nx.draw_networkx(poi_cs_graph, ax=ax2, pos=pos, with_labels=True,
                     labels=poi_cs_labels, node_color='r', font_color='w')
    nx.draw_networkx(new_cs_graph, ax=ax2, pos=pos, with_labels=False,
                     node_color='#00b4d9', font_color='w')

    # Сохранение изображения
    plt.savefig("result.png")


# ==================================Основная программа
if __name__ == '__main__':
    # сбор данных пользователя
    params = input_params()

    # формирование графа сетки для района города
    Net_Graph, pois_place, cs_place, csNewPotential_place = model_settings(params.width, params.height, params.poi,
                                                                           params.chargers)

    # формирование BQM
    bqm = form_modelBQM(csNewPotential_place, params.poi, pois_place, params.chargers, cs_place, params.new_chargers)

    # Запуск BQM
    sampler = LeapHybridSampler()
    print("\nЗапуск модели на ", sampler.solver.id, " решателе...")

    csNew_place = goBQM(bqm, sampler, csNewPotential_place)

    # Печать результатов
    printSolve(pois_place, params.poi, cs_place, params.chargers, csNew_place, params.new_chargers)

    # Сохранение модели в файл
    saveResults(Net_Graph, pois_place, cs_place, csNew_place)
