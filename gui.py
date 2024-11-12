import numpy as np
import pandas as pd
import pygame
import pygame_gui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Agent:
    COLORS = {
        'leader': (255, 0, 0),  # Red
        'follower': (0, 0, 255),  # Blue
        'explorer': (0, 255, 0),  # Green
        'neutral': (200, 200, 200)  # Light Gray
    }

    def __init__(self, position, velocity, persona, health=1.0):
        self.position = position
        self.velocity = velocity
        self.persona = persona
        self.color = Agent.COLORS[persona]
        self.trail = [position.copy()]
        self.tracing = True
        self.health = health
        self.score = 0
        self.penalty = 0
        self.learning_rate = 0.1
        self.neutral_time = 0
        self.role_switch_cooldown = 0  # Initialize the cooldown period for switching roles

    def update(self, alignment_strength, avg_velocity, cohesion_strength, target_position, separation_strength, neighbors, separation_threshold, obstacles, rewards):
        # Decrement the cooldown if greater than zero
        if self.role_switch_cooldown > 0:
            self.role_switch_cooldown -= 1

        # If the agent is neutral, allow them to recover after some time
        if self.persona == 'neutral':
            self.neutral_time += 1
            if self.neutral_time > 50:
                self.persona = np.random.choice(['leader', 'follower', 'explorer'])
                self.color = Agent.COLORS[self.persona]
                self.neutral_time = 0

        # Penalty logic that switches agent to neutral
        if self.penalty > 10:
            self.persona = 'neutral'
            self.color = Agent.COLORS[self.persona]
            alignment_strength *= (1 - self.learning_rate)
            cohesion_strength *= (1 - self.learning_rate)
            self.penalty = 0

        # Update behavior based on persona
        if self.persona == 'leader':
            cohesion_strength *= 1.5
            target_reward = self.find_closest_reward(rewards)
            if target_reward is not None:
                direction_to_reward = target_reward - self.position
                self.velocity += 0.1 * direction_to_reward
        elif self.persona == 'follower':
            alignment_strength *= 1.5
            target_reward = self.find_closest_reward(rewards)
            if target_reward is not None:
                direction_to_reward = target_reward - self.position
                self.velocity += 0.05 * direction_to_reward
        elif self.persona == 'explorer':
            self.velocity += (np.random.rand(2) - 0.5) * 0.5

        # General behaviors
        self.velocity += alignment_strength * (avg_velocity - self.velocity)
        direction_to_target = target_position - self.position
        self.velocity += cohesion_strength * direction_to_target

        # Separation behavior
        for neighbor in neighbors:
            diff = self.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist < separation_threshold and dist > 0:
                self.velocity += separation_strength * (diff / dist)

            if dist < 2.0:
                self.velocity = -self.velocity
                self.penalty += 1

        # Obstacle avoidance
        for obstacle in obstacles:
            diff = self.position - obstacle
            dist = np.linalg.norm(diff)
            if dist < 10.0 and dist > 0:
                self.velocity += 0.5 * (diff / dist)

        # Reward collection logic
        rewards_to_remove = []
        for i, reward in enumerate(rewards):
            if np.linalg.norm(self.position - reward) < 5.0:
                self.score += 1
                rewards_to_remove.append(i)

        # Remove rewards and add new ones (in reverse order to avoid index issues)
        for i in sorted(rewards_to_remove, reverse=True):
            rewards.pop(i)
            rewards.append(np.random.rand(2) * 100)

        # Role transition based on score (considering cooldown)
        if self.role_switch_cooldown == 0:
            if self.persona != 'leader' and self.score >= 10:
                self.persona = 'leader'
                self.color = Agent.COLORS['leader']
                self.role_switch_cooldown = 50  # Set cooldown period
            elif self.persona != 'follower' and self.score >= 5:
                self.persona = 'follower'
                self.color = Agent.COLORS['follower']
                self.role_switch_cooldown = 50
            elif self.persona == 'neutral' and self.score >= 2:
                self.persona = 'explorer'
                self.color = Agent.COLORS['explorer']
                self.role_switch_cooldown = 50

        # Normalize velocity
        max_speed = 2.0
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed

        # Update position and handle boundaries
        self.position += self.velocity
        self.position = np.clip(self.position, 5, 95)  # Ensure agents stay strictly within visible boundaries (5 to 95)

        if self.tracing:
            self.trail.append(self.position.copy())

    def find_closest_reward(self, rewards):
        if len(rewards) == 0:
            return None
        distances = [np.linalg.norm(self.position - reward) for reward in rewards]
        closest_index = np.argmin(distances)
        return rewards[closest_index]

class SwarmSimulation:
    def __init__(self, num_agents=50, alignment_strength=0.5, cohesion_strength=0.01, separation_strength=0.05, separation_threshold=5.0, neighbor_radius=20.0):
        self.num_agents = num_agents
        self.alignment_strength = alignment_strength
        self.cohesion_strength = cohesion_strength
        self.separation_strength = separation_strength
        self.separation_threshold = separation_threshold
        self.neighbor_radius = neighbor_radius
        personas = ['leader', 'follower', 'explorer', 'neutral']
        self.agents = [
            Agent(np.random.rand(2) * 100, (np.random.rand(2) - 0.5) * 10, np.random.choice(personas))
            for _ in range(num_agents)
        ]
        self.agent_data = pd.DataFrame(columns=['Step', 'AgentID', 'Position', 'Score', 'Penalty'])
        self.step = 0
        self.obstacles = [np.random.rand(2) * 100 for _ in range(10)]
        self.rewards = [np.random.rand(2) * 100 for _ in range(20)]
        self.persona_counts = {'leader': 0, 'follower': 0, 'explorer': 0, 'neutral': 0}
        self.update_persona_counts()
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 10))  # Create figure with 2 subplots
        plt.ion()  # Interactive mode on
        plt.show()

    def update(self):
        avg_velocity = np.mean([agent.velocity for agent in self.agents], axis=0)
        avg_position = np.mean([agent.position for agent in self.agents], axis=0)
        for i, agent in enumerate(self.agents):
            neighbors = [
                other_agent for other_agent in self.agents
                if np.linalg.norm(agent.position - other_agent.position) < self.neighbor_radius and agent != other_agent
            ]
            agent.update(self.alignment_strength, avg_velocity, self.cohesion_strength, avg_position, self.separation_strength, neighbors, self.separation_threshold, self.obstacles, self.rewards)
            self.agent_data = pd.concat([
                self.agent_data,
                pd.DataFrame({'Step': [self.step], 'AgentID': [i], 'Position': [agent.position.copy()], 'Score': [agent.score], 'Penalty': [agent.penalty]})
            ], ignore_index=True)
        self.update_persona_counts()
        self.step += 1
        self.visualize()

    def update_persona_counts(self):
        self.persona_counts = {'leader': 0, 'follower': 0, 'explorer': 0, 'neutral': 0}
        for agent in self.agents:
            self.persona_counts[agent.persona] += 1

    def get_positions_and_colors(self):
        return [(agent.position, agent.color) for agent in self.agents]

    def get_trails(self):
        return [agent.trail for agent in self.agents]

    def get_obstacles(self):
        return self.obstacles

    def get_rewards(self):
        return self.rewards

    def toggle_tracing(self):
        for agent in self.agents:
            agent.tracing = not agent.tracing

    def visualize(self):
        # Clear previous data in subplots
        self.axs[0].clear()
        self.axs[1].clear()

        # Visualize rewards collected by each agent
        scores = [agent.score for agent in self.agents]
        self.axs[0].bar(range(len(scores)), scores, color='blue')
        self.axs[0].set_xlabel('Agent ID')
        self.axs[0].set_ylabel('Rewards Collected')
        self.axs[0].set_title('Rewards Collected by Each Agent')

        # Visualize the number of agents in each role
        labels = list(self.persona_counts.keys())
        counts = list(self.persona_counts.values())
        self.axs[1].bar(labels, counts, color=['red', 'blue', 'green', 'gray'])
        self.axs[1].set_xlabel('Persona')
        self.axs[1].set_ylabel('Count')
        self.axs[1].set_title('Number of Agents in Each Role')

        plt.draw()
        plt.pause(0.001)

class SwarmApp:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 700
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Swarm Intelligence Simulator")
        self.simulation = SwarmSimulation()
        self.clock = pygame.time.Clock()
        self.running = True
        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height))

        self.alignment_slider = pygame_gui.elements.ui_horizontal_slider.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 10), (200, 30)), start_value=0.5, value_range=(0.0, 1.0), manager=self.manager, object_id='#alignment_slider')
        self.alignment_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 10), (150, 30)), text='Alignment Strength', manager=self.manager)

        self.cohesion_slider = pygame_gui.elements.ui_horizontal_slider.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 50), (200, 30)), start_value=0.01, value_range=(0.0, 0.1), manager=self.manager, object_id='#cohesion_slider')
        self.cohesion_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 50), (150, 30)), text='Cohesion Strength', manager=self.manager)

        self.separation_slider = pygame_gui.elements.ui_horizontal_slider.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 90), (200, 30)), start_value=0.05, value_range=(0.0, 0.1), manager=self.manager, object_id='#separation_slider')
        self.separation_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 90), (150, 30)), text='Separation Strength', manager=self.manager)

        self.separation_threshold_slider = pygame_gui.elements.ui_horizontal_slider.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 130), (200, 30)), start_value=5.0, value_range=(1.0, 20.0), manager=self.manager, object_id='#separation_threshold_slider')
        self.separation_threshold_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 130), (150, 30)), text='Separation Threshold', manager=self.manager)

        self.num_agents_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((10, 170), (200, 30)), manager=self.manager)
        self.num_agents_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((220, 170), (150, 30)), text='Number of Agents', manager=self.manager)
        self.num_agents_input.set_text(str(self.simulation.num_agents))

        self.toggle_tracing_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 210), (200, 50)), text='Toggle Tracing', manager=self.manager)

    def update_simulation(self):
        self.simulation.update()

    def draw_elements(self):
        self.screen.fill((0, 0, 0))
        for agent in self.simulation.agents:
            x, y = agent.position * (self.screen_width / 100)
            color = agent.color
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 5)
            if agent.tracing and len(agent.trail) > 1:
                trail_points = [(int(pos[0] * (self.screen_width / 100)), int(pos[1] * (self.screen_height / 100))) for pos in agent.trail]
                pygame.draw.lines(self.screen, color, False, trail_points, 1)
        for obstacle in self.simulation.get_obstacles():
            x, y = obstacle * (self.screen_width / 100)
            pygame.draw.circle(self.screen, (255, 0, 0), (int(x), int(y)), 10)
        for reward in self.simulation.get_rewards():
            x, y = reward * (self.screen_width / 100)
            pygame.draw.circle(self.screen, (255, 255, 0), (int(x), int(y)), 5)  # Changed reward color to yellow

        # Draw legend with counts
        font = pygame.font.Font(None, 24)
        legend_items = [
            ("Leader", Agent.COLORS['leader'], self.simulation.persona_counts['leader']),
            ("Follower", Agent.COLORS['follower'], self.simulation.persona_counts['follower']),
            ("Explorer", Agent.COLORS['explorer'], self.simulation.persona_counts['explorer']),
            ("Neutral", Agent.COLORS['neutral'], self.simulation.persona_counts['neutral']),
            ("Reward", (255, 255, 0), len(self.simulation.rewards))  # Added legend for rewards
        ]
        y_offset = 10
        for label, color, count in legend_items:
            text_surface = font.render(f"{label}: {count}", True, color)
            self.screen.blit(text_surface, (self.screen_width - 200, y_offset))
            y_offset += 30

    def handle_ui_events(self, event):
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.toggle_tracing_button:
                self.simulation.toggle_tracing()
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.alignment_slider:
                self.simulation.alignment_strength = event.value
            elif event.ui_element == self.cohesion_slider:
                self.simulation.cohesion_strength = event.value
            elif event.ui_element == self.separation_slider:
                self.simulation.separation_strength = event.value
            elif event.ui_element == self.separation_threshold_slider:
                self.simulation.separation_threshold = event.value
        elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element == self.num_agents_input:
                try:
                    new_num_agents = int(event.text)
                    if new_num_agents > 0:
                        self.simulation = SwarmSimulation(num_agents=new_num_agents)
                except ValueError:
                    pass

    def run(self):
        while self.running:
            time_delta = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                self.manager.process_events(event)
                self.handle_ui_events(event)

            self.update_simulation()
            self.manager.update(time_delta)
            self.draw_elements()
            self.manager.draw_ui(self.screen)

            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    app = SwarmApp()
    app.run()
