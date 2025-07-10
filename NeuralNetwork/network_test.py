from network import Network
import numpy as np

network = Network((1, 4, 1))

x = [(1), (2), (3), (4)]
y = [(2), (4), (6), (8)]

print("TRAINING DATA:")
print(f"Input: {x}")
print(f"Target: {y}")

network.sgd_train(x, y, 3000, 0.001)

print("Test inputs (stochastic):")
print(f"0.1 -> {network.result(0.1)}")
print(f"5 -> {network.result(5)}")
print(f"7 -> {network.result(7)}")
print(f"10 -> {network.result(10)}")
print(f"20 -> {network.result(20)}")
print(f"1000 -> {network.result(1000)}")

network = Network((1, 4, 1))
network.batch_train(x, y, 3, 1000, 0.01)

print("Test inputs (batch size = 3):")
print(f"0.1 -> {network.result(0.1)}")
print(f"5 -> {network.result(5)}")
print(f"7 -> {network.result(7)}")
print(f"10 -> {network.result(10)}")
print(f"20 -> {network.result(20)}")
print(f"1000 -> {network.result(1000)}")