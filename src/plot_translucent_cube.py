import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the vertices of a unit cube
vertices = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
]

# Define the 6 faces of the cube using the vertices
faces = [
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
    [vertices[3], vertices[0], vertices[4], vertices[7]],  # Left face
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
    [vertices[0], vertices[1], vertices[2], vertices[3]]   # Bottom face
]

# Add the cube to the plot with translucency
ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, edgecolor='k'))

# Set the limits of the plot
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
