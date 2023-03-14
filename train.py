from get_data import get_mnist_data
from model import EnsambleModel

def main():
    X, y = get_mnist_data()
    em = EnsambleModel()
    em.train(X, y)

if __name__ == "__main__":
    main()
