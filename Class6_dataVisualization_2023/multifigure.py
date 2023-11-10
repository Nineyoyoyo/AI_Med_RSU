import cv2
import matplotlib.pyplot as plt

def line_plot():
    x = [0,1,2,3,4,5]
    y = [4,4,2,5,1,5]

    plt.plot(x,y) 
    plt.show()


def test_imshow():
    img = cv2.imread('/Users/internmet/Desktop/nine_vscode/AI_med/Class6_dataVisualization_2023/images/test_imshow_image.jpg',cv2.IMREAD_GRAYSCALE)
    plt.imshow(img,cmap='binary_r')
    plt.show()
    
    
def test_multi_figure():
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 100)

    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(2 * x)

    plt.figure()
    plt.plot(x, y1, label='sin(x)', color='red')

    plt.figure()
    plt.plot(x, y2, label='cos(x)', color='blue')

    plt.figure()
    plt.scatter(x,y3)

    plt.show()

    
if __name__=='__main__':
    # line_plot()
    # test_imshow()
    # test_multi_figure()
    