import random
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '32'


def get_simulation_result():
    count = 0
    for i in range(100):
        if random.random() > 0.5:
            count += 1
    return count


if __name__ == '__main__':
    # for _ in range(10):
    #     print(get_simulation_result())

    count_list = []
    for _ in range(1000000):
        count = get_simulation_result()
        count_list.append(count)

    M = sum(count_list) / len(count_list)
    print('M', M)

    plt.hist(count_list, bins=100, range=(0, 100))
    plt.show()


