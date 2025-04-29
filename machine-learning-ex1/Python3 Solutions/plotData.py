import matplotlib.pyplot as plt

def plot_data(X, y):
    plt.figure(figsize=(8,6))
    plt.plot(X, y, 'rx', markersize=10)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Training Data')
    plt.grid(True)
    plt.show()