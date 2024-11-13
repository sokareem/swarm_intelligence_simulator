import numpy as np
import pandas as pd
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame_gui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.widgets import Button, Slider
from collections import deque
import time

class Agent:
    COLORS = {
        'leader': (1.0, 0.0, 0.0),    # Red
        'follower': (0.0, 0.0, 1.0),  # Blue
        'explorer': (0.0, 1.0, 0.0),  # Green
        'neutral': (0.8, 0.8, 0.8)    # Light Gray
    }

    def __init__(self, position, velocity, persona, health=1.0):
        self.position = position
        self.velocity = velocity
        self.persona = persona
        self.color = Agent.COLORS[persona]
        self.trail = deque(maxlen=50)  # Limit trail length for better performance
        self.trail.append(position.copy())
        self.tracing = True
        self.health = health
        self.score = 0
        self.penalty = 0
        self.learning_rate = 0.1
        self.neutral_time = 0
        self.role_switch_cooldown = 0
        
        # Create display list for sphere
        self.sphere_dl = glGenLists(1)
        glNewList(self.sphere_dl, GL_COMPILE)
        sphere = gluNewQuadric()
        gluQuadricNormals(sphere, GLU_SMOOTH)  # Enable smooth normals
        gluSphere(sphere, 1.0, 16, 16)
        gluDeleteQuadric(sphere)
        glEndList()

    def update(self, alignment_strength, avg_velocity, cohesion_strength, target_position, 
              separation_strength, neighbors, separation_threshold, obstacles, rewards):
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
            self.velocity += (np.random.rand(3) - 0.5) * 0.5  # Now 3D random movement

        # General behaviors in 3D
        self.velocity += alignment_strength * (avg_velocity - self.velocity)
        direction_to_target = target_position - self.position
        self.velocity += cohesion_strength * direction_to_target

        # 3D Separation behavior
        for neighbor in neighbors:
            diff = self.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist < separation_threshold and dist > 0:
                self.velocity += separation_strength * (diff / dist)

            if dist < 2.0:
                self.velocity = -self.velocity
                self.penalty += 1

        # 3D Obstacle avoidance
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

        # Remove rewards and add new ones
        for i in sorted(rewards_to_remove, reverse=True):
            rewards.pop(i)
            rewards.append(np.random.rand(3) * 100)  # 3D reward positions

        # Role transition logic
        if self.role_switch_cooldown == 0:
            if self.persona != 'leader' and self.score >= 10:
                self.persona = 'leader'
                self.color = Agent.COLORS['leader']
                self.role_switch_cooldown = 50
            elif self.persona != 'follower' and self.score >= 5:
                self.persona = 'follower'
                self.color = Agent.COLORS['follower']
                self.role_switch_cooldown = 50
            elif self.persona == 'neutral' and self.score >= 2:
                self.persona = 'explorer'
                self.color = Agent.COLORS['explorer']
                self.role_switch_cooldown = 50

        # Normalize velocity in 3D
        max_speed = 2.0
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed

        # Update position and handle 3D boundaries
        self.position += self.velocity
        self.position = np.clip(self.position, 5, 95)

        if self.tracing:
            self.trail.append(self.position.copy())

    def find_closest_reward(self, rewards):
        if len(rewards) == 0:
            return None
        distances = [np.linalg.norm(self.position - reward) for reward in rewards]
        closest_index = np.argmin(distances)
        return rewards[closest_index]


    def __del__(self):
        # Clean up display lists when agent is destroyed
        try:
            glDeleteLists(self.sphere_dl, 1)
        except:
            pass

    def draw(self):
        glPushMatrix()
        glTranslatef(*self.position)
        glColor3f(*self.color)
        glCallList(self.sphere_dl)
        
        if self.tracing and len(self.trail) > 1:
            glDisable(GL_LIGHTING)
            glBegin(GL_LINE_STRIP)
            glColor3f(*[c * 0.7 for c in self.color])  # Slightly darker trail
            for pos in self.trail:
                glVertex3f(*pos)
            glEnd()
            glEnable(GL_LIGHTING)
        glPopMatrix()

class SwarmSimulation3D:
    """
    A class representing a 3D simulation of swarm intelligence.
    
    This class is responsible for managing and updating the state of agents in a 3D environment.
    """
    def __init__(self, num_agents=50):
        self.num_agents = num_agents
        self.alignment_strength = 0.3
        self.cohesion_strength = 0.01
        self.separation_strength = 0.05
        self.separation_threshold = 5.0
        self.neighbor_radius = 20.0
        
        personas = ['leader', 'follower', 'explorer', 'neutral']
        self.agents = [
            Agent(np.random.rand(3) * 100, (np.random.rand(3) - 0.5) * 10, 
                  np.random.choice(personas))
            for _ in range(num_agents)
        ]
        
        self.agent_data = pd.DataFrame(columns=['Step', 'AgentID', 'Position', 'Score', 'Penalty'])
        self.step = 0
        self.fps_samples = deque(maxlen=60)
        self.last_time = time.time()
        self.obstacles = [np.random.rand(3) * 100 for _ in range(10)]
        self.rewards = [np.random.rand(3) * 100 for _ in range(20)]
        self.persona_counts = {'leader': 0, 'follower': 0, 'explorer': 0, 'neutral': 0}
        self.update_persona_counts()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))  # Create figure with 3 subplots
        plt.ion()  # Interactive mode on
        plt.show()
        
        # Create display list for coordinate axes
        self.setup_display_lists()

    def setup_display_lists(self):
        self.axes_dl = glGenLists(1)
        glNewList(self.axes_dl, GL_COMPILE)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(100, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 100, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 100)
        glEnd()
        glEndList()

    def update(self):
        avg_velocity = np.mean([agent.velocity for agent in self.agents], axis=0)
        avg_position = np.mean([agent.position for agent in self.agents], axis=0)
        
        for i, agent in enumerate(self.agents):
            neighbors = [
                other_agent for other_agent in self.agents
                if np.linalg.norm(agent.position - other_agent.position) < self.neighbor_radius 
                and agent != other_agent
            ]
            agent.update(self.alignment_strength, avg_velocity, self.cohesion_strength, 
                        avg_position, self.separation_strength, neighbors, 
                        self.separation_threshold, self.obstacles, self.rewards)
            
            self.agent_data = pd.concat([
                self.agent_data,
                pd.DataFrame({
                    'Step': [self.step], 
                    'AgentID': [i], 
                    'Position': [agent.position.copy()],
                    'Score': [agent.score], 
                    'Penalty': [agent.penalty]
                })
            ], ignore_index=True)
            
        self.update_persona_counts()
        self.step += 1
        self.visualize()

    def __del__(self):
        # Clean up display lists
        try:
            glDeleteLists(self.axes_dl, 1)
        except:
            pass

    def update_persona_counts(self):
        self.persona_counts = {'leader': 0, 'follower': 0, 'explorer': 0, 'neutral': 0}
        for agent in self.agents:
            self.persona_counts[agent.persona] += 1

    def visualize(self):
        # Clear previous data in subplots
        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()

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

        # Visualize positions of obstacles and rewards
        obstacles_positions = np.array(self.obstacles)
        rewards_positions = np.array(self.rewards)
        if len(obstacles_positions) > 0:
            self.axs[2].scatter(obstacles_positions[:, 0], obstacles_positions[:, 1], c='red', label='Obstacles')
        if len(rewards_positions) > 0:
            self.axs[2].scatter(rewards_positions[:, 0], rewards_positions[:, 1], c='yellow', label='Rewards')
        self.axs[2].set_xlim([0, 100])
        self.axs[2].set_ylim([0, 100])
        self.axs[2].set_xlabel('X Position')
        self.axs[2].set_ylabel('Y Position')
        self.axs[2].set_title('Obstacles and Rewards Positions')
        self.axs[2].legend()

        plt.draw()
        plt.pause(0.001)

    def draw_obstacles_and_rewards(self):
        # Draw obstacles
        glColor3f(1.0, 0.0, 0.0)  # Red for obstacles
        for obstacle in self.obstacles:
            glPushMatrix()
            glTranslatef(*obstacle)
            glutSolidSphere(2.0, 16, 16)  # Draw obstacle as a sphere
            glPopMatrix()

        # Draw rewards
        glColor3f(1.0, 1.0, 0.0)  # Yellow for rewards
        for reward in self.rewards:
            glPushMatrix()
            glTranslatef(*reward)
            glutSolidSphere(1.5, 16, 16)  # Draw reward as a smaller sphere
            glPopMatrix()

class SwarmApp3D:
    def __init__(self):
        pygame.init()
        self.screen_width = 1024
        self.screen_height = 768
        
        # Set OpenGL attributes before creating the window
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)  # Enable VSync
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 
                                            DOUBLEBUF | OPENGL | pygame.HWSURFACE)
        pygame.display.set_caption("3D Swarm Intelligence Simulator")
        
        self.simulation = SwarmSimulation3D()
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps_font = pygame.font.Font(None, 36)

        # Create the Matplotlib UI for controlling parameters
        self.create_controller_panel()

        
        # Initialize UI manager
        self.ui_manager = pygame_gui.UIManager((self.screen_width, self.screen_height))
        self.persona_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 80), (300, 30)),
            text='Personas: Leader, Follower, Explorer, Neutral',
            manager=self.ui_manager
        )
        self.fps_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (150, 30)),
            text='FPS: 0',
            manager=self.ui_manager
        )
        self.num_agents_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 50), (150, 30)),
            text=f'Agents: {self.simulation.num_agents}',
            manager=self.ui_manager
        )
    #     self.alignment_slider = pygame_gui.elements.ui_horizontal_slider.UIHorizontalSlider(
    #     relative_rect=pygame.Rect((10, 120), (200, 30)),
    #     start_value=self.simulation.alignment_strength,
    #     value_range=(0.0, 1.0),
    #     manager=self.ui_manager,
    #     object_id='#alignment_slider'
    # )

    #     self.separation_threshold_slider = pygame_gui.elements.ui_horizontal_slider.UIHorizontalSlider(
    #         relative_rect=pygame.Rect((10, 160), (200, 30)),
    #         start_value=self.simulation.separation_threshold,
    #         value_range=(1.0, 20.0),
    #         manager=self.ui_manager,
    #         object_id='#separation_threshold_slider'
    # )  
    #     self.alignment_label = pygame_gui.elements.UILabel(
    #         relative_rect=pygame.Rect((220, 120), (150, 30)),
    #         text='Alignment Strength',
    #         manager=self.ui_manager
    #     )

    #     self.separation_threshold_label = pygame_gui.elements.UILabel(
    #         relative_rect=pygame.Rect((220, 160), (150, 30)),
    #         text='Separation Threshold',
    #         manager=self.ui_manager
    #     )
    
        
        # Initialize OpenGL
        self.setup_gl()
        
        # Performance monitoring
        self.frame_times = deque(maxlen=60)
        self.last_time = time.time()
        
        # Camera settings
        self.camera_distance = 200
        self.camera_rotation_h = 0
        self.camera_rotation_v = 30
        self.camera_target = np.array([50, 50, 50])

    
    def create_controller_panel(self):
        # Create a separate figure for the controller UI
        self.controller_fig, self.controller_axs = plt.subplots(5, 1, figsize=(5, 10))
        plt.subplots_adjust(left=0.2, bottom=0.5)  # Adjust to fit sliders with extra room for all sliders
        plt.ion()  # Interactive mode on

        # Slider for Alignment Strength
        self.alignment_slider_ax = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.alignment_slider = Slider(self.alignment_slider_ax, 'Alignment', 0.0, 1.0, valinit=self.simulation.alignment_strength)
        
        # Slider for Cohesion Strength
        self.cohesion_slider_ax = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.cohesion_slider = Slider(self.cohesion_slider_ax, 'Cohesion', 0.0, 0.1, valinit=self.simulation.cohesion_strength)

        # Slider for Separation Threshold
        self.separation_slider_ax = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.separation_slider = Slider(self.separation_slider_ax, 'Separation Threshold', 1.0, 20.0, valinit=self.simulation.separation_threshold)

        # Slider for Number of Agents
        self.num_agents_slider_ax = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.num_agents_slider = Slider(self.num_agents_slider_ax, 'Num Agents', 10, 100, valinit=self.simulation.num_agents, valstep=1)

        # Slider for Number of Rewards
        self.num_rewards_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.num_rewards_slider = Slider(self.num_rewards_slider_ax, 'Num Rewards', 1, 50, valinit=len(self.simulation.rewards), valstep=1)

        # Button to toggle trails on and off
        self.toggle_trails_button_ax = plt.axes([0.4, 0.075, 0.2, 0.05])
        self.toggle_trails_button = Button(self.toggle_trails_button_ax, 'Toggle Trails')

        # Button to apply agent/reward changes
        self.update_button_ax = plt.axes([0.4, 0.025, 0.2, 0.05])
        self.update_button = Button(self.update_button_ax, 'Update')

        # Connect the sliders and button to their update functions
        self.alignment_slider.on_changed(self.update_alignment_strength)
        self.cohesion_slider.on_changed(self.update_cohesion_strength)
        self.separation_slider.on_changed(self.update_separation_threshold)
        self.update_button.on_clicked(self.update_num_agents_and_rewards)
        self.toggle_trails_button.on_clicked(self.toggle_trails)

        plt.show()

    def update_alignment_strength(self, val):
        self.simulation.alignment_strength = val

    def update_cohesion_strength(self, val):
        self.simulation.cohesion_strength = val

    def update_separation_threshold(self, val):
        self.simulation.separation_threshold = val

    def update_num_agents_and_rewards(self, event):
        # Get updated number of agents from slider
        new_num_agents = int(self.num_agents_slider.val)
        if new_num_agents != self.simulation.num_agents:
            # Update the agents list in the simulation
            personas = ['leader', 'follower', 'explorer', 'neutral']
            self.simulation.agents = [
                Agent(np.random.rand(3) * 100, (np.random.rand(3) - 0.5) * 10, 
                      np.random.choice(personas))
                for _ in range(new_num_agents)
            ]
            self.simulation.num_agents = new_num_agents

        # Get updated number of rewards from slider
        new_num_rewards = int(self.num_rewards_slider.val)
        if new_num_rewards != len(self.simulation.rewards):
            # Update the rewards list in the simulation
            self.simulation.rewards = [np.random.rand(3) * 100 for _ in range(new_num_rewards)]
    
    def toggle_trails(self, event):
        # Toggle the tracing feature for all agents in the simulation
        for agent in self.simulation.agents:
            agent.tracing = not agent.tracing



    def setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)  # Use smooth shading
        
        # Better lighting setup
        glLightfv(GL_LIGHT0, GL_POSITION, (100.0, 100.0, 100.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        
        # Material properties
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)
        
        # Enable face culling for better performance
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Enable antialiasing
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glEnable(GL_MULTISAMPLE)
        
        # Set clear color to dark gray
        glClearColor(0.2, 0.2, 0.2, 1.0)


    def render_text(self, text, x, y, color=(255, 255, 255)):
        # Switch to 2D rendering for text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting and depth test for text
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Render text
        text_surface = self.fps_font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glRasterPos2i(x, y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                    GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Restore previous state
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_3d_scene(self):
        # Update FPS calculation
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        fps = len(self.frame_times) / sum(self.frame_times)
        
        # Clear and setup view
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.screen_width / self.screen_height), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Camera positioning
        eye_pos = np.array([
            self.camera_distance * np.sin(np.radians(self.camera_rotation_h)) * np.cos(np.radians(self.camera_rotation_v)),
            self.camera_distance * np.sin(np.radians(self.camera_rotation_v)),
            self.camera_distance * np.cos(np.radians(self.camera_rotation_h)) * np.cos(np.radians(self.camera_rotation_v))
        ]) + self.camera_target
        
        gluLookAt(*eye_pos, *self.camera_target, 0, 1, 0)
        
        # Draw coordinate system
        glCallList(self.simulation.axes_dl)
        
        # Draw agents
        for agent in self.simulation.agents:
            agent.draw()

        # TODO: Draw Obstacles and rewards
        # [Insert Here]
        self.simulation.draw_obstacles_and_rewards()
        
        # Draw performance info
        self.render_text(f"FPS: {fps:.1f}", 10, 10)
        self.render_text(f"Agents: {len(self.simulation.agents)}", 10, 40)
        
        # Draw persona counts
        y_offset = 70
        for persona, count in self.simulation.persona_counts.items():
            color = [int(c * 255) for c in Agent.COLORS[persona]]
            self.render_text(f"{persona.capitalize()}: {count}", 10, y_offset, color)
            y_offset += 30

        # Update UI labels
        self.fps_label.set_text(f'FPS: {int(fps)}')
        self.num_agents_label.set_text(f'Agents: {len(self.simulation.agents)}')
        
    def run(self):
        try:
            while self.running:
                time_delta = self.clock.tick(60) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                        self.camera_rotation_h += event.rel[0] * 0.5
                        self.camera_rotation_v = np.clip(
                            self.camera_rotation_v - event.rel[1] * 0.5,
                            -89, 89
                        )
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 4:  # Mouse wheel up
                            self.camera_distance = max(50, self.camera_distance - 10)
                        elif event.button == 5:  # Mouse wheel down
                            self.camera_distance = min(500, self.camera_distance + 10)

                    # Process UI events to check if sliders have moved
                    self.ui_manager.process_events(event)
                    if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                        if event.ui_element == self.alignment_slider:
                            self.simulation.alignment_strength = event.value
                        elif event.ui_element == self.separation_threshold_slider:
                            self.simulation.separation_threshold = event.value

                self.simulation.update()
                self.draw_3d_scene()
                self.ui_manager.update(time_delta)
                self.ui_manager.draw_ui(self.screen)
                pygame.display.flip()
        finally:
            pygame.quit()

if __name__ == "__main__":
    app = SwarmApp3D()
    app.run()
