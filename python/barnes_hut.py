import numpy as np

# Define a class to represent each particle
class Body:
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.force = np.zeros(2)  # Force accumulator (2D vector)

    def reset_force(self):
        self.force = np.zeros(2)

    def add_force(self, other):
        # Calculate the force exerted by another particle using Newton's law of gravitation
        G = 6.67430e-11  # Gravitational constant
        eps = 1e-3  # Softening factor to avoid singularities
        diff = other.pos - self.pos
        dist = np.linalg.norm(diff)
        force_magnitude = (G * self.mass * other.mass) / (dist**2 + eps**2)
        self.force += force_magnitude * diff / (dist + eps)

    def update(self, dt):
        # Update position and velocity using the force
        self.vel += self.force * dt / self.mass
        self.pos += self.vel * dt

# Define a class to represent the quadtree node
class QuadTreeNode:
    def __init__(self, center, size):
        self.center = np.array(center, dtype=float)  # Center of the node
        self.size = size  # Size (width) of the node
        self.body = None  # Contains a single body (if leaf)
        self.total_mass = 0.0  # Total mass in the region
        self.center_of_mass = np.zeros(2)  # Center of mass of all bodies
        self.children = [None] * 4  # Four quadrants (for 2D)

    def insert(self, body):
        if self.body is None and all(child is None for child in self.children):
            # If the node is empty and no children, put the body here
            self.body = body
            self.total_mass = body.mass
            self.center_of_mass = body.pos
        else:
            # If already contains a body, subdivide and distribute the bodies
            if self.body:
                # Subdivide and distribute the current body
                self.subdivide()
                self._insert_into_quadrant(self.body)
                self.body = None  # Clear body because this is no longer a leaf
            # Insert the new body into the correct quadrant
            self._insert_into_quadrant(body)
            # Update total mass and center of mass
            self.total_mass += body.mass
            self.center_of_mass = (self.center_of_mass * (self.total_mass - body.mass) + body.mass * body.pos) / self.total_mass

    def _insert_into_quadrant(self, body):
        # Determine which quadrant the body belongs to
        quadrant_index = (body.pos[0] > self.center[0]) + 2 * (body.pos[1] > self.center[1])
        if not self.children[quadrant_index]:
            new_center = self.center.copy()
            offset = self.size / 4
            if quadrant_index & 1:
                new_center[0] += offset
            else:
                new_center[0] -= offset
            if quadrant_index & 2:
                new_center[1] += offset
            else:
                new_center[1] -= offset
            self.children[quadrant_index] = QuadTreeNode(new_center, self.size / 2)
        self.children[quadrant_index].insert(body)

    def subdivide(self):
        # Subdivide the current node into four quadrants
        for i in range(4):
            new_center = self.center.copy()
            offset = self.size / 4
            if i & 1:
                new_center[0] += offset
            else:
                new_center[0] -= offset
            if i & 2:
                new_center[1] += offset
            else:
                new_center[1] -= offset
            self.children[i] = QuadTreeNode(new_center, self.size / 2)

    def compute_force(self, body, theta=0.5):
        # If this is an external node, calculate force directly
        if all(child is None for child in self.children):
            if self.body and self.body != body:
                body.add_force(self.body)
        else:
            # Otherwise, apply the Barnes-Hut approximation
            distance = np.linalg.norm(self.center_of_mass - body.pos)
            if self.size / distance < theta:
                # If far enough, approximate by treating the entire region as a single mass
                dummy_body = Body(self.total_mass, self.center_of_mass, np.zeros(2))
                body.add_force(dummy_body)
            else:
                # Otherwise, recursively compute forces from each child node
                for child in self.children:
                    if child:
                        child.compute_force(body, theta)

# Define a function to simulate one time step using the Barnes-Hut algorithm
def simulate_barnes_hut(bodies, bounds, dt, theta=0.5):
    # Create the quadtree root node (covers the entire simulation area)
    root = QuadTreeNode(center=(bounds[0] / 2, bounds[1] / 2), size=max(bounds))
    # Insert all bodies into the quadtree
    for body in bodies:
        root.insert(body)

    # Compute forces for each body
    for body in bodies:
        body.reset_force()
        root.compute_force(body, theta)

    # Update the positions and velocities of the bodies
    for body in bodies:
        body.update(dt)

# Example usage:
if __name__ == "__main__":
    # Create some particles
    bodies = [
        Body(1e5, [0.5, 0.5], [0.0, 0.0]),
        Body(1e5, [1.5, 0.5], [0.0, 0.1]),
        Body(1e5, [0.5, 1.5], [-0.1, 0.0]),
    ]

    # Simulation parameters
    bounds = [2.0, 2.0]  # Size of the simulation area
    dt = 0.01  # Time step
    num_steps = 1000

    # Run the simulation
    for step in range(num_steps):
        simulate_barnes_hut(bodies, bounds, dt)

        # Print the positions of the bodies at each step
        for i, body in enumerate(bodies):
            print(f"Step {step}, Body {i}, Pos: {body.pos}, Vel: {body.vel}")
