#! /bin/python3

import matplotlib.pyplot as plt


def load_data(filename: str):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            row = line.split(',')
            record = dict()
            record['method'] = row[2]
            record['game_count'] = int(row[3])
            record['time'] = int(row[4]) + int(row[5]) + int(row[6])
            data += [record]

    d = dict()
    for record in data:
        if record['method'] in d:
            if record['game_count'] in d[record['method']]:
                d[record['method']][record['game_count']]['time'] += record['time']
                d[record['method']][record['game_count']]['count'] += 1
            else:
                d[record['method']][record['game_count']] = {'time': record['time'], 'count': 1}
        else:
            d[record['method']] = {record['game_count']: {'time': record['time'], 'count': 1}}

    data = d

    d = dict()
    for method, val in data.items():
        d[method] = {'counts': [], 'times': []}
        for game_count, v in val.items():
            d[method]['counts'] += [game_count]
            d[method]['times'] += [v['time'] / v['count']]

    return d


def plot_data(method: str, data):
    plt.legend()
    plt.plot(data[method]['counts'], data[method]['times'], label=method)


data = load_data('log_time')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('time (ns)')
plt.xlabel('game_count')

plot_data('cpu', data)
plot_data('run_simulation_step_0', data)
plot_data('run_simulation_step_1', data)

plt.legend()
plt.show()

