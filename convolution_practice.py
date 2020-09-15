import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import misc

def plotImg(img):
    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def main():

    # Load the image into the variable i
    i = misc.ascent()

    # Plot the image
    plotImg(i)

    # Copy the image
    i_transformed = np.copy(i)

    # Pull the size of the image
    size_x = i_transformed.shape[0]
    size_y = i_transformed.shape[1]

    # Begin the convolution process for the image

    # Define a filter
    filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]               #
    # filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]             #
    # filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]               #

    # Set a weight, this depends on if the sum of the values within a filter do not equal 1
    weight = 1

    # Iterate through and calculate the convolution for every pixel in the original image
    for x in range(1, size_x - 1):
        for y in range(1, size_y - 1):
            convolution = 0.0

            # Find the convolution by iterating through the neighbors
            convolution = convolution + ( i[x - 1, y - 1] * filter[0][0] )
            convolution = convolution + ( i[x, y - 1] * filter[0][1] )
            convolution = convolution + ( i[x + 1, y - 1] * filter[0][2] )
            convolution = convolution + ( i[x - 1, y] * filter[1][0] )
            convolution = convolution + ( i[x, y] * filter[1][1] )
            convolution = convolution + ( i[x + 1, y] * filter[1][2] )
            convolution = convolution + ( i[x - 1, y + 1] * filter[2][0] )
            convolution = convolution + ( i[x, y + 1] * filter[2][1] )
            convolution = convolution + ( i[x + 1, y + 1] * filter[2][2] )

            # Normalize the convolution
            convolution = convolution * weight

            if(convolution < 0):
                convolution = 0
            elif(convolution > 255):
                convolution = 255

            # Set the convolution
            i_transformed[x, y] = convolution

    # Plot the transformed image
    plotImg(i_transformed)

    # Begin the Max Pooling Process

    # New image size
    new_x = int(size_x/2)
    new_y = int(size_y/2)

    # Initialize new image
    newImage = np.zeros((new_x, new_y))

    # Iterate through the image
    for x in range(0, size_x, 2):
        for y in range(0, size_y, 2):

            # Determine the max pixel out of every set of 4 pixels
            pixels = []
            pixels.append(i_transformed[x, y])
            pixels.append(i_transformed[x + 1, y])
            pixels.append(i_transformed[x, y + 1])
            pixels.append(i_transformed[x + 1, y + 1])
            pixels.sort(reverse = True)
            newImage[int(x/2), int(y/2)] = pixels[0]

    # Plot the Pooled image
    plotImg(newImage)

if __name__ == "__main__":
    main()