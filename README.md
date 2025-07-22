# Swarm Intelligence Simulator

This Swarm Intelligence Simulator is an interactive environment that models the behavior of agents exhibiting swarm-like dynamics. It allows you to visualize the behavior of various agents ("swarm members") who interact with each other and their environment to achieve specific roles, such as leading, exploring, or following. The simulator is built using Python, Pygame, and Matplotlib for both the graphical user interface (GUI) and visualization.

## Features

- **Agent Roles**: Agents can take on different roles within the swarm: leader, follower, explorer, or neutral. Each role affects how agents behave, move, and interact.
- **Interactive GUI**: The simulator provides a user-friendly graphical interface with adjustable sliders and buttons to manipulate the swarm's parameters in real-time. These controls include alignment strength, cohesion strength, separation strength, and separation threshold.
- **Dynamic Role Changes**: Agents can change roles during the simulation based on their accumulated rewards and penalties. Their roles are represented with different colors to help distinguish behaviors:
  - **Leader** (Red)
  - **Follower** (Blue)
  - **Explorer** (Green)
  - **Neutral** (Light Gray)
- **Rewards and Penalties**: Agents can collect rewards (represented by yellow circles) and avoid obstacles (represented by red circles). Depending on their performance, they can earn points or penalties that may change their role within the swarm.
- **Real-time Visualizations**: The simulator includes real-time visualizations of the rewards collected by each agent, as well as the number of agents in each role. These visualizations update continuously, allowing you to observe the impact of parameter changes on swarm behavior.

## How It Works

The simulator models a 2D space where agents move around based on simple rules:

1. **Alignment**: Agents adjust their velocity to match the average velocity of neighboring agents, allowing the swarm to stay cohesive.
2. **Cohesion**: Agents are attracted to the average position of their neighbors, which keeps the swarm together.
3. **Separation**: Agents avoid getting too close to their neighbors, preventing collisions.
4. **Role-Specific Behavior**:
   - **Leaders** actively seek out rewards and have an increased cohesion effect.
   - **Followers** adjust their alignment strength to stick closer to the group and move towards rewards with a lower priority than leaders.
   - **Explorers** move more randomly to explore new areas in the environment.
   - **Neutral** agents have diminished abilities and must regain their roles by avoiding penalties and accumulating enough rewards.

### Boundaries and Constraints

Agents operate within a bounded environment to ensure they remain visible on the screen. Positions are clamped within a range of 5 to 95 units to prevent them from lingering at the borders. This helps in maintaining a consistent and meaningful simulation without agents going off-screen.

### Reward and Role Transition Logic

- Agents can collect rewards when they move close enough to a reward point. Once collected, the reward is removed from the screen and a new one is generated at a random location.
- Based on their accumulated score, agents can transition into new roles. For example:
  - If an agent's score reaches 10, they become a leader.
  - If their score reaches 5, they become a follower.
  - If they receive multiple penalties, they can lose their role and become neutral.
- A cooldown period is also in place for role transitions to avoid erratic behavior and ensure smooth transitions between roles.

### Interactive GUI Elements

The GUI includes various controls to adjust simulation parameters:
- **Alignment Strength Slider**: Controls how strongly agents align with their neighbors.
- **Cohesion Strength Slider**: Controls how strongly agents are attracted to the average position of their neighbors.
- **Separation Strength Slider**: Controls how strongly agents avoid their neighbors.
- **Separation Threshold Slider**: Adjusts the distance at which agents start avoiding their neighbors.
- **Number of Agents Input**: Allows the user to change the number of agents in the simulation.
- **Toggle Tracing Button**: Enables or disables the trail tracing feature for each agent, which visually indicates the path an agent has taken.

## Getting Started

### Prerequisites
- Python 3.10+
- Required Libraries: `pygame`, `pygame_gui`, `numpy`, `matplotlib`, `pandas`

### Installation
1. Clone this repository.
2. Install the required libraries using pip:
   ```sh
   pip install pygame pygame_gui numpy matplotlib pandas
   ```
3. Run the simulation:
   ```sh
   python gui.py
   ```

## Running the Simulation
Upon running the script, the simulation window will open, displaying the agents in action. Use the sliders to adjust parameters in real-time and observe how the agents adapt their behaviors.

### Visualizing Data
Real-time visualizations of the rewards collected and agent roles are displayed alongside the simulation to provide insight into how agents are performing.

## Future Enhancements
- Implement reinforcement learning to enable agents to learn optimal behaviors over time.
- Add more complex environments, such as maze-like obstacles or dynamic reward placements.
- Introduce different types of rewards and penalties to diversify agent behavior.

## License
This project is licensed under the MIT License.

# Ant Colony Optimization Simulation (Reinforcement Learning & Adversarial Dynamics)

This project is an advanced Ant Colony Optimization (ACO) simulation that incorporates Reinforcement Learning (RL) and adversarial dynamics. It models the behavior of two competing ant colonies, each striving for dominance and resource control. The simulation provides a dynamic and visually rich environment to observe emergent intelligence and inter-colony conflict.

## Features

- **Reinforcement Learning (DQN)**: Ants learn optimal foraging and survival strategies using a Deep Q-Network (DQN). They adapt their behavior based on rewards and penalties, leading to intelligent decision-making.
- **Adversarial Nests**: Two distinct ant colonies (Blue and Red) compete for shared food resources and territory. Each colony has its own nest and agents.
- **Emergent Evolution (Warrior Ants)**: Colonies can unlock a new class of specialized ants, the "Warrior Ants," by accumulating sufficient food resources. Warrior Ants possess enhanced combat capabilities.
- **Dynamic Visuals**:
  - **Fading Ant Trails**: Ants leave subtle, fading trails that visually represent pheromone paths and their recent movements.
  - **Dynamic Ant Appearance**: Ant colors subtly change based on their energy levels, providing visual cues to their vitality.
  - **Pulsating Nests**: Nests glow with an intensity proportional to their stored food, reflecting the colony's prosperity.
  - **Food Deposit Cues**: A visual flash at the nest indicates successful food deposits, highlighting the purpose of foraging.
  - **Warrior Unlocked Indicators**: Nests display a visual marker when Warrior Ants have been unlocked.
- **Interactive Elements**: Users can dynamically add new food sources (left-click) and obstacles (right-click) to the environment, allowing for real-time experimentation and observation of the swarm's adaptation.
- **Inter-Colony Conflict**: Ants from opposing colonies engage in direct combat, losing energy upon collision. Ants with zero energy are eliminated.
- **Victory Conditions**: The simulation concludes when one colony achieves dominance, either by eliminating all opposing ants or by demonstrating overwhelming resource superiority while the opponent is depleted. This signifies the "completion of the great work."

## How It Works

The simulation models a 2D environment where ants from two colonies interact with each other, food sources, and obstacles. Key mechanisms include:

- **Pheromone System**: Ants deposit and sense pheromones to guide their foraging paths. Pheromones evaporate over time, reinforcing efficient routes.
- **Energy Management**: Ants consume energy for movement and actions. They must return to their nest to replenish energy by consuming stored food.
- **DQN-driven Behavior**: Each ant's actions (moving towards nest/food, scouting, avoiding obstacles, staying) are determined by a trained DQN, which learns to maximize rewards.
- **Specialized Roles**: Ants can be Workers (foraging), Caretakers (supporting low-energy ants), or Warriors (combat-focused).
- **Dynamic Environment**: Obstacles periodically reset, forcing continuous adaptation from the colonies.

## Getting Started

### Prerequisites
- Python 3.10+
- Required Libraries: `pygame`, `pygame_gui`, `numpy`, `matplotlib`, `pandas`, `torch`

### Installation
1. Clone this repository.
2. Create and activate a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required libraries:
   ```sh
   pip install pygame pygame_gui numpy matplotlib pandas torch
   ```

## Running the Simulation

To run the advanced Ant Colony Optimization simulation:

```sh
venv/bin/python3 ant_colony_nn.py
```

Upon running the script, the simulation window will open, displaying the two competing ant colonies. Use mouse clicks to interact with the environment:
- **Left-click**: Add a new food source.
- **Right-click**: Add a new obstacle.

Observe how the colonies adapt, forage, and engage in conflict. Watch for the emergence of Warrior Ants and the visual cues indicating nest prosperity and food deposits.

## Future Enhancements
- Implement more sophisticated inter-colony combat mechanics.
- Explore different RL algorithms for ant behavior.
- Introduce environmental factors like weather or terrain.
- Develop a more complex research tree for new ant classes.
- Add a graphical display for real-time learning metrics (loss, reward).

## License
This project is licensed under the MIT License.

