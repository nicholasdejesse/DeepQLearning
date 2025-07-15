from network import Network
import numpy as np

# ONE INPUT, ONE OUTPUT
x = [(1), (2), (4), (10), (20)]
y = [(2), (4), (8), (20), (40)]

print("TRAINING DATA:")
print(f"Input: {x}")
print(f"Target: {y}")

network = Network((1, 4, 1))
network.train(x, y, 1, 3000, 0.0005)

print("Test inputs (stochastic):")
print(f"(50) -> {network.result((50))}")

network = Network((1, 4, 1))
network.train(x, y, 3, 3000, 0.001)

print("Test inputs (batch size = 3):")
print(f"(50) -> {network.result((50))}")


# ONE INPUT, TWO OUTPUTS
x = [(4), (6), (8), (12), (20)]
y = [(2, 2), (3, 3), (4, 4), (6, 6), (10, 10)]

print("TRAINING DATA:")
print(f"Input: {x}")
print(f"Target: {y}")

network = Network((1, 4, 2))
network.train(x, y, 1, 1000, 0.001)

print("Test inputs (stochastic):")
print(f"(14) -> {network.result((14))}")

network = Network((1, 4, 2))
network.train(x, y, 3, 1000, 0.001)

print("Test inputs (batch size = 3):")
print(f"(14) -> {network.result((14))}")


# TWO INPUTS, TWO OUTPUTS
x = [(1, 3), (5, 2), (9, 18), (32, 4), (18, 19)]
y = [(2, 2), (6, 1), (10, 17), (33, 3), (19, 18)]

print("TRAINING DATA:")
print(f"Input: {x}")
print(f"Target: {y}")

network = Network((2, 4, 2))
network.train(x, y, 1, 1000, 0.001)

print("Test inputs (stochastic):")
print(f"(14, 17) -> {network.result((14, 17))}")

network = Network((2, 4, 2))
network.train(x, y, 3, 1000, 0.001)

print("Test inputs (batch size = 3):")
print(f"(14, 17) -> {network.result((14, 17))}")
