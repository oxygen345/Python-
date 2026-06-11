# 导入库
from random import seed
from random import randrange


def train_test_split(dataset, train=0.6):
    """
    划分训练集与测试集

    dataset：原始数据集
    train：训练集比例，默认 0.6
    """

    # 创建训练集
    train_basket = list()

    # 根据输入的训练集比例计算训练集大小
    train_size = train * len(dataset)

    # 复制一份原始数据集，避免改变原数据
    dataset_copy = list(dataset)

    # 循环抽取数据，直到训练集达到指定大小
    while len(train_basket) < train_size:
        # 随机生成训练集索引
        random_choose = randrange(len(dataset_copy))

        # 根据索引取出样本，加入训练集
        # pop 会把该样本从 dataset_copy 中删除，避免重复抽取
        train_basket.append(dataset_copy.pop(random_choose))

    # 返回训练集和测试集
    return train_basket, dataset_copy


# 主函数
if __name__ == '__main__':

    # 设置随机种子，保证每次运行结果一样
    seed(666)

    # 创建数据集
    dataset = [
        [1], [2], [3], [4], [5],
        [6], [7], [8], [9], [10]
    ]

    # 调用手动编写的 train_test_split 函数划分训练集和测试集
    train, test = train_test_split(dataset)

    print('训练集：')
    print(train)

    print('测试集：')
    print(test)