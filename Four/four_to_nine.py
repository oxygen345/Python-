# 导入库
from random import seed
from random import randrange


def k_fold_cross_validation_split(dataset, folds=10):
    """
    该函数用于将数据集执行K折交叉验证的划分

    dataset：二维列表，传入需要划分交叉验证的数据集
    folds：整数，可选，传入交叉验证的折数，默认为10

    返回参数 basket_split_data：三维列表，存放的是划分好的交叉验证的数据集
    """

    # 定义一个空列表，用于存放划分好的数据集
    basket_split_data = list()

    # 计算每一折里的样本数
    fold_size = int(len(dataset) / folds)

    # 复制出一个新的数据集来做划分，从而不改变原始数据集
    dataset_copy = list(dataset)

    # 循环生成每一折的数据
    for i in range(folds):

        # 定义一个空列表，用于存放每一折里的样本数
        basket_random_fold = list()

        # 开始遍历，只要每一折里的样本数小于 fold_size，就一直往里面添加随机数据
        while len(basket_random_fold) < fold_size:

            # 通过 randrange 函数随机产生索引
            random_choose_index = randrange(len(dataset_copy))

            # 根据随机索引将数据集中的样本加入到 basket_random_fold 中
            basket_random_fold.append(dataset_copy.pop(random_choose_index))

        # 每一折的样本数添加好后，再将其加入 basket_split_data
        basket_split_data.append(basket_random_fold)

    return basket_split_data


# 主函数
if __name__ == '__main__':

    # 定义一个随机种子，使得每次生成的随机数都是确定的
    seed(1)

    dataset = [
        [1], [2], [3], [4], [5],
        [6], [7], [8], [9], [10]
    ]

    # 调用手动编写的 k_fold_cross_validation_split 函数
    # 实现 K 折交叉验证数据集的划分
    k_folds_split = k_fold_cross_validation_split(dataset, 3)

    print(k_folds_split)