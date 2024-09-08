import numpy as np

# Define constants
G = 6.67430e-11  # Gravitational constant

# Define a class to represent each particle (or body)
class Body:
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.force = np.zeros(2)  # 2D force vector

    def reset_force(self):
        self.force = np.zeros(2)

    def update(self, dt):
        # Update position and velocity based on force
        self.vel += self.force * dt / self.mass
        self.pos += self.vel * dt

# Multipole Expansion for a cell in the tree
class MultipoleExpansion:
    def __init__(self):
        self.total_mass = 0.0
        self.center_of_mass = np.zeros(2)
        self.multipole = np.zeros(4)  # Quadrupole expansion (simplified)

    def add_body(self, body):
        # Add a body to the multipole expansion
        self.total_mass += body.mass
        self.center_of_mass += body.mass * body.pos

    def finalize(self):
        # Normalize center of mass
        if self.total_mass > 0:
            self.center_of_mass /= self.total_mass

# Define a class to represent a node in the FMM tree
class FMMNode:
    def __init__(self, center, size):
        self.center = np.array(center, dtype=float)
        self.size = size
        self.bodies = []
        self.children = None  # This will hold the child nodes (4 for 2D)
        self.multipole = MultipoleExpansion()

    def insert(self, body):
        # Insert a body into the node
        if len(self.bodies) < 1 and self.children is None:
            # If it's a leaf node, add the body
            self.bodies.append(body)
        else:
            if self.children is None:
                # Subdivide the node if necessary
                self.subdivide()
            # Insert the body into the appropriate child
            self.insert_into_child(body)
        self.multipole.add_body(body)

    def subdivide(self):
        # Subdivide the node into 4 quadrants (2D case)
        half_size = self.size / 2
        offset = half_size / 2
        self.children = [
            FMMNode(self.center + [-offset, -offset], half_size),
            FMMNode(self.center + [offset, -offset], half_size),
            FMMNode(self.center + [-offset, offset], half_size),
            FMMNode(self.center + [offset, offset], half_size),
        ]
        # Distribute current bodies among the children
        for body in self.bodies:
            self.insert_into_child(body)
        self.bodies = []  # Clear the bodies from this node

    def insert_into_child(self, body):
        # Find the appropriate quadrant for the body and insert it
        quadrant = (body.pos[0] > self.center[0]) + 2 * (body.pos[1] > self.center[1])
        self.children[quadrant].insert(body)

    def finalize(self):
        # Finalize the multipole expansion after all bodies are inserted
        if self.children is None:
            self.multipole.finalize()
        else:
            for child in self.children:
                child.finalize()

    def compute_force(self, body, theta=0.5):
        # Compute force on the body using the multipole approximation
        if self.children is None:
            # Direct interaction for small nodes (leaf)
            for other_body in self.bodies:
                if body != other_body:
                    body.force += self.compute_direct_force(body, other_body)
        else:
            # Approximate force if the cell is far enough
            dist = np.linalg.norm(body.pos - self.multipole.center_of_mass)
            if self.size / dist < theta:
                # Apply multipole approximation
                body.force += self.multipole_approximation(body)
            else:
                # Otherwise, recurse into children
                for child in self.children:
                    child.compute_force(body, theta)

    def compute_direct_force(self, body1, body2):
        # Calculate the direct gravitational force between two bodies
        eps = 1e-3  # Softening factor to avoid singularities
        diff = body2.pos - body1.pos
        dist = np.linalg.norm(diff)
        if dist == 0:
            return np.zeros(2)
        force_magnitude = (G * body1.mass * body2.mass) / (dist**2 + eps**2)
        return force_magnitude * diff / (dist + eps)

    def multipole_approximation(self, body):
        # Compute the force from the multipole expansion of the node
        eps = 1e-3  # Softening factor
        diff = self.multipole.center_of_mass - body.pos
        dist = np.linalg.norm(diff)
        if dist == 0:
            return np.zeros(2)
        force_magnitude = (G * body.mass * self.multipole.total_mass) / (dist**2 + eps**2)
        return force_magnitude * diff / (dist + eps)

# Define a function to simulate one time step using FMM
def simulate_fmm(bodies, bounds, dt, theta=0.5):
    # Create the FMM tree root node (covers the entire simulation area)
    root = FMMNode(center=(bounds[0] / 2, bounds[1] / 2), size=max(bounds))
    
    # Insert all bodies into the FMM tree
    for body in bodies:
        root.insert(body)
    
    # Finalize the tree (compute multipole expansions)
    root.finalize()
    
    # Compute forces for each body
    for body in bodies:
        body.reset_force()
        root.compute_force(body, theta)
    
    # Update the positions and velocities of the bodies
    for body in bodies:
        body.update(dt)

# Example usage:
if __name__ == "__main__":
    # Create some bodies (particles)
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
        simulate_fmm(bodies, bounds, dt)

        # Print the positions of the bodies at each step
        for i, body in enumerate(bodies):
            print(f"Step {step}, Body {i}, Pos: {body.pos}, Vel: {body.vel}")
