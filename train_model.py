from src.data_loader import load_data
from src.train import train

if __name__ == '__main__':
    fabrics, outfits = load_data('data/fabfashion.csv')
    if len(fabrics) == 0 or len(outfits) == 0:
        print("\u274c No valid image pairs found. Please check your CSV paths or image folders.")
        exit()

    train(fabrics, outfits, epochs=100)