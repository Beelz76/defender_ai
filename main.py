import pygame
from src.game import Defender
from src.agent import DefenderDQNAgent
from src.ui import GameUI


def main():
    pygame.init()
    game = Defender()
    agent = DefenderDQNAgent(game)
    ui = GameUI(game, agent)
    ui.run()


if __name__ == "__main__":
    main()
