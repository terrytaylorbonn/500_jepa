#14a_open_npz.py
from numpy import load

data = load('./data/toy2d_14a_data.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])