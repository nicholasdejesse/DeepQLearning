from network import Network
import numpy as np

network = Network((2, 4, 1))

x = [(2, 3), (4, 7), (5, 3), (9, 6), (1, 2), (4, 4)]
y = [(5), (11), (8), (15), (3), (8)]

print("TRAINING DATA:")
print(f"Input: {x}")
print(f"Target: {y}")

network.train(x, y, 1, 3000, 0.001)

print("Test inputs (stochastic):")
print(f"(10, 10) -> {network.result((10, 10))}")
# print(f"0.1 -> {network.result(0.1)}")
# print(f"5 -> {network.result(5)}")
# print(f"7 -> {network.result(7)}")
# print(f"10 -> {network.result(10)}")
# print(f"20 -> {network.result(20)}")
# print(f"1000 -> {network.result(1000)}")

network = Network((2, 4, 1))
network.train(x, y, 3, 3000, 0.001)

print("Test inputs (batch size = 3):")
print(f"(10, 10) -> {network.result((10, 10))}")
# print(f"0.1 -> {network.result(0.1)}")
# print(f"5 -> {network.result(5)}")
# print(f"7 -> {network.result(7)}")
# print(f"10 -> {network.result(10)}")
# print(f"20 -> {network.result(20)}")
# print(f"1000 -> {network.result(1000)}")