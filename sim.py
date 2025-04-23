import random
import math
import statistics
import matplotlib.pyplot as plt

def generate_random_points_in_sphere(N):
    points = []
    while len(points) < N:
        x, y, z = [random.uniform(-1, 1) for _ in range(3)]
        if x**2 + y**2 + z**2 <= 1:
            points.append((x, y, z))
    return points

def average_pairwise_distance(points):
    total_distance = 0
    count = 0
    N = len(points)
    
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            total_distance += distance
            count += 1

    return total_distance / count

def compute_average_distance_for_N(N, R_g=14_000_000_000):
    points = generate_random_points_in_sphere(N)
    avg = average_pairwise_distance(points)
    return avg * R_g

N_values = list(range(100, 1100, 100))
average_distances = []

for N in N_values:
    print(f"Computing for N = {N}...")
    scaled_avg = compute_average_distance_for_N(N)
    average_distances.append(scaled_avg)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(N_values, average_distances, marker='o')
plt.title('Average Pairwise Distance vs. Number of Galaxies')
plt.xlabel('Number of Galaxies (N)')
plt.ylabel('Average Distance (light years)')
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_distance_vs_N.png")
plt.show()