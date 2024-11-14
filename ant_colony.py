import random
import numpy as np
import pygame

# Pheromone map class
class Pheromone:
    def __init__(self, size=100, evaporation_rate=0.01):
        self.grid = np.zeros((size, size))
        self.evaporation_rate = evaporation_rate
        self.size = size

    def deposit(self, position, amount):
        x, y = position.astype(int)
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[x, y] += amount

    def evaporate(self):
        self.grid *= (1 - self.evaporation_rate)

    def get_concentration(self, position):
        x, y = position.astype(int)
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        else:
            return 0

# Nest class
class Nest:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.food_stored = 200
        self.contributions = {}
        self.pheromone_strength = 1.0
        self.food_consumed = 0  # New  to track total food consumed by ants

    def record_contribution(self, agent_id, amount):
        if agent_id not in self.contributions:
            self.contributions[agent_id] = 0
        self.contributions[agent_id] += amount
        self.food_stored += amount
        self.food_consumed += amount

    def consume_food(self, amount):
        # Define a minimum food threshold
        min_food_threshold = 100
        
        # Only allow consumption if food stored after consumption is above the threshold
        if self.food_stored - amount >= min_food_threshold:
            self.food_stored -= amount
            self.food_consumed += amount
            
            return amount
        elif self.food_stored > min_food_threshold:
            food_available = self.food_stored - min_food_threshold
            self.food_stored -= food_available
            self.food_consumed += food_available
            
            return food_available
        else:
            # Not enough food to consume
            return 0


# Agent (Ant) class
class Agent:
    STATES = {
        'foraging': (0, 255, 0),    # Green
        'returning': (255, 165, 0),  # Orange
        'scouting': (255, 0, 0),    # Red
        'resting': (200, 200, 200)   # Gray
    }

    def __init__(self, position, nest, pheromone_map, role='worker'):
        self.position = np.array(position, dtype=float)
        angle = np.random.uniform(0, 2*np.pi)
        self.velocity = np.array([np.cos(angle), np.sin(angle)])
        self.state = 'scouting'
        self.role = role  # 'worker' or 'caretaker'
        self.color = (0, 0, 255) if self.role == 'caretaker' else Agent.STATES[self.state]
        self.carrying_food = 0
        if role == 'caretaker':
            self.energy = 500
            self.max_energy = 500
        else:
            self.energy = 200
            self.max_energy = 200
        self.dead = False
        self.nest = nest
        self.agent_id = id(self)
        self.pheromone_map = pheromone_map
        self.memory = []
        self.last_food_position = None
        self.rest_time = 0
        # Attributes for forage timeout
        self.forage_time = 0
        self.forage_threshold = 300  # Adjust as needed

    def update(self, food_sources, obstacles, all_agents):
       # Energy consumption
        self.energy -= 0.01  # Reduced from 0.0125
        if self.energy <= 0:
            self.dead = True
            return

        # Efficient energy management
        if self.energy <= 20 and self.state not in ['returning', 'resting']:
            self.state = 'returning'
            self.color = Agent.STATES[self.state]

        # Resting logic
        if self.state == 'resting':
                self.rest_time += 1
                # Ant consumes food from the nest to replenish energy
                energy_needed = self.max_energy - self.energy
                food_available = self.nest.consume_food(energy_needed)
                self.energy += food_available
                if self.energy > self.max_energy:
                    self.energy = self.max_energy

                if self.rest_time >= 50 or self.energy >= self.max_energy:
                    if self.last_food_position is not None:
                        self.state = 'foraging'  # Resume foraging if possible
                    else:
                        self.state = 'scouting'
                    self.color = Agent.STATES[self.state]
                    self.rest_time = 0
                return


        # State machine logic
        if self.state == 'returning':
            self.return_to_nest()
        elif self.state == 'foraging':
            self.follow_pheromone_trail()
        elif self.state == 'scouting':
            self.scout()

        # Obstacle avoidance
        self.avoid_obstacles(obstacles)

        # Food collection logic
        if not self.carrying_food:
            self.check_for_food(food_sources)

        # Deposit pheromones if carrying food
        if self.carrying_food:
            self.deposit_pheromone()

        # Update position
        self.move()

        # Check if at nest
        if np.linalg.norm(self.position - self.nest.position) < 3.0:
            if self.carrying_food:
                self.deposit_food()
            elif self.energy <= 20:
                self.state = 'resting'
                self.color = Agent.STATES[self.state]

        # Check if last_food_position is invalid
        if self.last_food_position is not None:
            if np.linalg.norm(self.position - self.last_food_position) < 3.0:
                # Check if there is any food at last_food_position
                food_found = False
                for food in food_sources:
                    if np.array_equal(food['position'], self.last_food_position) and food['quantity'] > 0:
                        food_found = True
                        break
                if not food_found:
                    # No food at last_food_position
                    self.last_food_position = None
                    self.state = 'scouting'
                    self.color = Agent.STATES[self.state]

        # Additional logic for forage timeout
        if self.state == 'foraging':
            self.forage_time += 1
            if self.forage_time >= self.forage_threshold:
                # Switch to scouting after exceeding forage threshold
                self.state = 'scouting'
                self.color = Agent.STATES[self.state]
                self.forage_time = 0  # Reset forage_time
           # Caretaker-specific behavior
        if self.role == 'caretaker':
            self.caretaker_behavior(all_agents)

    def caretaker_behavior(self, all_agents):
        # Identify the nearest agent with low energy
        target_agent = self.find_low_energy_agent(all_agents)
        if target_agent:
            # Move towards the target agent
            self.move_towards(target_agent.position)
            # If close enough, transfer energy
            if np.linalg.norm(self.position - target_agent.position) < 3.0:
                self.transfer_energy(target_agent)

    def find_low_energy_agent(self, all_agents):
        # Define energy threshold for assistance
        energy_threshold = 20
        # Find agents (excluding caretakers) with energy below the threshold
        low_energy_agents = [agent for agent in all_agents 
                             if agent != self and agent.energy < energy_threshold and agent.role != 'caretaker']
        if low_energy_agents:
            # Find the nearest low-energy agent
            distances = [np.linalg.norm(self.position - agent.position) for agent in low_energy_agents]
            min_distance = min(distances)
            target_agent = low_energy_agents[distances.index(min_distance)]
            return target_agent
        return None

    def move_towards(self, target_position):
        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            self.velocity = direction / distance

    def transfer_energy(self, target_agent):
        transfer_amount = 20  # Define how much energy to transfer
        if self.energy > transfer_amount + 20:  # Ensure caretaker retains minimum energy
            target_agent.energy += transfer_amount
            self.energy -= transfer_amount
            print(f"Caretaker {self.agent_id} transferred {transfer_amount} energy to Ant {target_agent.agent_id}")
        else:
            # Not enough energy to transfer
            pass

    def deposit_food(self):
        # Ant consumes a portion of the food carried to replenish energy
        energy_needed = self.max_energy - self.energy
        energy_from_food = min(self.carrying_food, energy_needed)
        self.energy += energy_from_food
        self.carrying_food -= energy_from_food

        # Deposit any remaining food to the nest
        self.nest.record_contribution(self.agent_id, self.carrying_food)
        self.carrying_food = 0
        self.memory = []
        if self.last_food_position is not None:
            self.state = 'foraging'
            self.color = (0, 0, 255) if self.role == 'caretaker' else Agent.STATES[self.state]
        else:
            self.state = 'scouting'
            self.color = (0, 0, 255) if self.role == 'caretaker' else Agent.STATES[self.state]

    def scout(self):
        # Random walk
        angle = np.random.uniform(0, 2*np.pi)
        self.velocity = np.array([np.cos(angle), np.sin(angle)])

    def follow_pheromone_trail(self):
        # Sample pheromone concentrations around the ant
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        directions = np.array([np.cos(angles), np.sin(angles)]).T
        sample_points = self.position + directions * 2
        concentrations = np.array([self.pheromone_map.get_concentration(p) for p in sample_points])

        # Probabilistic movement based on pheromone concentration
        if concentrations.sum() > 0:
            probabilities = concentrations / concentrations.sum()
            chosen_direction = directions[np.random.choice(len(directions), p=probabilities)]
            self.velocity = chosen_direction
        else:
            # No pheromone detected
            if self.last_food_position is not None:
                # Move towards last known food position
                direction = self.last_food_position - self.position
                if np.linalg.norm(direction) > 0:
                    self.velocity = direction / np.linalg.norm(direction)
            else:
                # Perform random walk
                self.scout()

    def return_to_nest(self):
        direction = self.nest.position - self.position
        if np.linalg.norm(direction) > 0:
            self.velocity = direction / np.linalg.norm(direction)

    def avoid_obstacles(self, obstacles):
        for obstacle in obstacles:
            diff = self.position - obstacle
            dist = np.linalg.norm(diff)
            if dist < 5.0 and dist > 0:
                self.velocity += (diff / dist) * 0.5

    def check_for_food(self, food_sources):
        for food in food_sources:
            if np.linalg.norm(self.position - food['position']) < 3.0 and food['quantity'] > 0:
                # Ant can carry up to 5 units of food
                carry_amount = min(5, food['quantity'])
                self.carrying_food = carry_amount
                food['quantity'] -= carry_amount
                self.nest.record_contribution(self.agent_id, self.carrying_food)
                self.last_food_position = food['position'].copy()
                self.state = 'returning'
                self.color = Agent.STATES[self.state]
                self.forage_time = 0 # Reset forage time after finding food
                return

    def deposit_pheromone(self):
        # Deposit more pheromone if the path is shorter
        amount = 1.0 / (len(self.memory) + 1)
        self.pheromone_map.deposit(self.position, amount)

    def move(self):
        # Normalize velocity
        speed = np.linalg.norm(self.velocity)
        if speed > 2.0:
            self.velocity = (self.velocity / speed) * 2.0

        # Update position and handle boundaries
        self.position += self.velocity
        self.position = np.clip(self.position, 0, 99)

        # Record position in memory
        self.memory.append(self.position.copy())
        if len(self.memory) > 100:
            self.memory.pop(0)

# Swarm simulation class
class SwarmSimulation:
    def __init__(self, num_agents=50):
        self.initial_num_agents = num_agents  # Store initial population size
        self.num_agents = num_agents
        self.nest = Nest(position=[50, 50])
        self.pheromone_map = Pheromone()
        # Assign 10% of agents as caretakers
        num_caretakers = int(0.1 * num_agents)
        self.agents = [
            Agent(np.array(self.nest.position, dtype=float), self.nest, self.pheromone_map, role='caretaker') 
            if i < num_caretakers else 
            Agent(np.array(self.nest.position, dtype=float), self.nest, self.pheromone_map)
            for i in range(num_agents)
        ]
        self.obstacles = [np.random.rand(2) * 100 for _ in range(10)]
        self.food_sources = [{'position': np.random.rand(2) * 100, 'quantity': 100} for _ in range(10)]  # Increased to 10 food sources with 100 units each
        self.state_counts = {state: 0 for state in Agent.STATES.keys()}
        self.total_food_collected = []

    def update(self):
         # Update pheromone map
        self.pheromone_map.evaporate()

        # Update agents
        alive_agents = []
        for agent in self.agents:
            agent.update(self.food_sources, self.obstacles,self.agents)
            if not agent.dead:
                alive_agents.append(agent)
            else:
                print(f"Ant {agent.agent_id} has died.")
        self.agents = alive_agents  # Update the agents list

        # Ant birth mechanism
        self.ant_birth()

        # Update state counts
        self.update_state_counts()

        # Track total food collected
        self.total_food_collected.append(self.nest.food_stored)
    
    def ant_birth(self):
        # Define normal birth parameters
        normal_birth_threshold = 50
        normal_birth_energy = 2

        # Define enhanced birth parameters when population is low
        enhanced_birth_threshold = 10
        enhanced_birth_energy = 1

        # Determine current birth parameters based on population
        if len(self.agents) < (self.initial_num_agents / 2):
            birth_threshold = enhanced_birth_threshold
            birth_energy = enhanced_birth_energy
            print("Population below half. Enhanced ant birth parameters activated.")
        else:
            birth_threshold = normal_birth_threshold
            birth_energy = normal_birth_energy

        max_ants = 50  # Maximum number of ants allowed

        # While the food stored exceeds the threshold and we have not reached max ants
        while self.nest.food_stored >= birth_threshold and len(self.agents) < max_ants:
            # Consume food for birth
            self.nest.food_stored -= birth_energy
            # Randomly assign role with 50% chance
            role = random.choice(['worker', 'caretaker'])
            # Create a new ant at the nest with the assigned role
            new_ant = Agent(np.array(self.nest.position, dtype=float), self.nest, self.pheromone_map, role=role)
            self.agents.append(new_ant)
            print(f"A new {role} ant is born!")

    def update_state_counts(self):
        self.state_counts = {state: 0 for state in Agent.STATES.keys()}
        for agent in self.agents:
            self.state_counts[agent.state] += 1

    def add_food_source(self):
        # Remove depleted food sources and add new ones
        for food in self.food_sources:
            if food['quantity'] <= 0:
                food['position'] = np.random.rand(2) * 100
                food['quantity'] = 100

# SwarmApp class with Pygame visualization
class SwarmApp:
    def __init__(self):
        pygame.init()
        self.screen_width = 600
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Ant Colony Optimization Simulation")
        self.clock = pygame.time.Clock()
        self.simulation = SwarmSimulation(num_agents=50)  # Set initial agents to 100
        self.running = True

    def run(self):
        while self.running:
            self.handle_events()
            self.simulation.update()
            self.simulation.add_food_source()
            self.draw_elements()
            pygame.display.flip()
            self.clock.tick(60)  # Limit to 60 FPS

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def draw_elements(self):
        self.screen.fill((0, 0, 0))
        
        # Draw pheromone trails
        grid_size = self.simulation.pheromone_map.size  # Grid size is 100
        pheromone_surface = pygame.Surface((grid_size, grid_size))
        pheromone_surface.set_alpha(128)
        grid = self.simulation.pheromone_map.grid
        max_pheromone = grid.max()
        if max_pheromone > 0:
            # Normalize and scale pheromone values to [0, 255]
            scaled_grid = (grid / max_pheromone) * 255
            scaled_grid = scaled_grid.astype(np.uint8)
            
            # Create an array for the surface
            surface_array = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
            surface_array[:, :, 2] = scaled_grid  # Assign to the blue channel
            
            # Transpose the array to match Pygame's surface format
            surface_array = np.transpose(surface_array, (1, 0, 2))
            
            # Blit the array onto the pheromone surface
            pygame.surfarray.blit_array(pheromone_surface, surface_array)
            
            # Scale the pheromone surface to the screen size
            pheromone_surface = pygame.transform.scale(pheromone_surface, (self.screen_width, self.screen_height))
            self.screen.blit(pheromone_surface, (0, 0))
        
        # Draw agents
        for agent in self.simulation.agents:
            x, y = agent.position * (self.screen_width / 100)
            pygame.draw.circle(self.screen, agent.color, (int(x), int(y)), 3)
        
        # Draw food sources
        for food in self.simulation.food_sources:
            x, y = food['position'] * (self.screen_width / 100)
            size = int(5 * (food['quantity'] / 50))
            size = max(size, 2)  # Minimum size to ensure visibility
            pygame.draw.circle(self.screen, (255, 255, 0), (int(x), int(y)), size)
        
        # Draw nest
        nest_x, nest_y = self.simulation.nest.position * (self.screen_width / 100)
        pygame.draw.circle(self.screen, (139, 69, 19), (int(nest_x), int(nest_y)), 8)
        
        # Display total food stored
        font = pygame.font.SysFont(None, 24)
        text = font.render(f'Total Food Stored: {int(self.simulation.nest.food_stored)}', True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        # Display agent state counts
        y_offset = 30
        for state, count in self.simulation.state_counts.items():
            state_color = Agent.STATES[state]
            text = font.render(f'{state.capitalize()}: {count}', True, state_color)
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

        # Display ant population
        ant_population_text = font.render(f'Ant Population: {len(self.simulation.agents)}', True, (255, 255, 255))
        self.screen.blit(ant_population_text, (10, y_offset))
        y_offset += 20

        # Display total food consumed
        food_consumed_text = font.render(f'Total Food Consumed: {int(self.simulation.nest.food_consumed)}', True, (255, 255, 255))
        self.screen.blit(food_consumed_text, (10, y_offset))
        y_offset += 20

# Main execution
if __name__ == "__main__":
    app = SwarmApp()
    app.run()