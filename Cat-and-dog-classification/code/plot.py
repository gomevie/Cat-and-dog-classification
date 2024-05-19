# coding=utf-8


import matplotlib.pyplot as plt


Loss_list = []  #存储每次epoch损失值

def draw_fig(name, train, val, epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.title('loss' + ' train  vs  val', fontsize=20)
        plt.plot(x1, train)
        plt.plot(x1, val)
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.grid()
        plt.legend(['train', 'val'])
        plt.savefig("../lossAndacc/" +  "loss.png")
        plt.show()

    elif name =="acc":
        plt.cla()
        plt.title('acc' + ' train  vs  val', fontsize=20)
        plt.plot(x1, train)
        plt.plot(x1, val)
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('acc', fontsize=20)
        plt.grid()
        plt.legend(['train', 'val'])
        plt.savefig("../lossAndacc/"  + "acc.png")
        plt.show()


import random

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list



if __name__ == '__main__':
    train = random_int_list(0, 10, 50)
    val = random_int_list(0, 10, 50)
    draw_fig('acc', train, val, 50)