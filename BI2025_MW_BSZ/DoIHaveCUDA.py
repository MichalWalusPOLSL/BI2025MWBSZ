import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA jest dostępne.")
        device = torch.device("cuda")
    else:
        print("CUDA nie jest dostępne. Używam CPU.")
        device = torch.device("cpu")

    print(f"Wersja PyTorch: {torch.__version__}")