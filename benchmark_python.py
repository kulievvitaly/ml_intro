import time


def f_list():
    l = []
    for i in range(10**7):
        l.append((-1)**i*i**2)
    l.sort()


def f_dict():
    d = {}
    for i in range(10**7):
        d[i] = i ** 2

    for j in range(10**6):
        if j in d:
            d[j] += 10000 - j


def f_str():
    l = []
    for i in range(10 ** 7):
        l.append(str(i))
    result = '|'.join(l)
    length = len(result)


def f_sum():
    s = 0.0
    for i in range(10 ** 8):
        s += i * 0.001


'''
ubuntu 22.04
cpu 11800h


'''

if __name__ == '__main__':
    elapsed_list = []
    for _ in range(10):
        timer = time.time()
        f_list()
        elapsed = time.time() - timer
        print('elapsed %.3f seconds' % elapsed)
        elapsed_list.append(elapsed)

    print('elapsed average %.3f seconds' % (sum(elapsed_list) / len(elapsed_list)))


    # elapsed_list = []
    # for _ in range(10):
    #     timer = time.time()
    #     f_dict()
    #     elapsed = time.time() - timer
    #     print('elapsed %.3f seconds' % elapsed)
    #     elapsed_list.append(elapsed)
    #
    # print('elapsed average %.3f seconds' % (sum(elapsed_list) / len(elapsed_list)))


    # elapsed_list = []
    # for _ in range(10):
    #     timer = time.time()
    #     f_str()
    #     elapsed = time.time() - timer
    #     print('elapsed %.3f seconds' % elapsed)
    #     elapsed_list.append(elapsed)
    #
    # print('elapsed average %.3f seconds' % (sum(elapsed_list) / len(elapsed_list)))

    # elapsed_list = []
    # for _ in range(10):
    #     timer = time.time()
    #     f_sum()
    #     elapsed = time.time() - timer
    #     print('elapsed %.3f seconds' % elapsed)
    #     elapsed_list.append(elapsed)
    #
    # print('elapsed average %.3f seconds' % (sum(elapsed_list) / len(elapsed_list)))