import numpy as np
import pygame
import sys
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

pygame.init()
screen = pygame.display.set_mode((600, 400))

# Fonts
OPEN_SANS = "OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)

pixels = [[0] * 28 for _ in range(28)]
classification = None

while True:

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            sys.exit()

    screen.fill((0, 0, 0))

    # Check for mouse press
    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # Draw each grid cell
    cells = []
    for i in range(28):
        row = []
        for j in range(28):
            cell = pygame.Rect(
                20 + j * 10,
                20 + i * 10,
                10, 10
            )

            # Darken cell if written on
            if pixels[i][j]:
                channel = 255 - (pixels[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), cell)

            # draw blank cell
            else:
                pygame.draw.rect(screen, (255, 255, 255), cell)
            pygame.draw.rect(screen, (0, 0, 0), cell, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse and cell.collidepoint(mouse):
                pixels[i][j] = 250 / 255
                if i + 1 < 28:
                    pixels[i + 1][j] = 220 / 255
                if j + 1 < 28:
                    pixels[i][j + 1] = 220 / 255
                if i + 1 < 28 and j + 1 < 28:
                    pixels[i + 1][j + 1] = 190 / 255

    # Reset button
    resetButton = pygame.Rect(
        # 20: offset
        # 28: number rows
        # 10: cell size
        30, 20 + 28 * 10 + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, (0, 0, 0))
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, (255, 255, 255), resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify button
    classifyButton = pygame.Rect(
        150, 20 + 28 * 10 + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, (255, 255, 255))
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, (0, 0, 0), classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # Reset drawing
    if mouse and resetButton.collidepoint(mouse):
        pixels = [[0] * 28 for _ in range(28)]
        classification = None

    # Generate classification
    if mouse and classifyButton.collidepoint(mouse):
        classification = model.predict(
            [np.array(pixels).reshape(1, 28, 28, 1)]
        ).argmax()

    # Show classification if one exists
    if classification is not None:
        classificationText = largeFont.render(str(classification), True, (255, 255, 255))
        classificationRect = classificationText.get_rect()
        grid_size = 20 * 20 + 10 * 28
        classificationRect.center = (
            grid_size + ((28 - grid_size) / 2),
            100
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()
