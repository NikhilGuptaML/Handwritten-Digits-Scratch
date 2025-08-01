import urllib.request, os

os.makedirs("data", exist_ok=True)   # create folder if it doesn't exist

urls = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
]


for url in urls:
    filename = os.path.join("data", url.split("/")[-1])
    print(f"Downloading {url} → {filename}")
    urllib.request.urlretrieve(url, filename)

print("All files downloaded into /data")
