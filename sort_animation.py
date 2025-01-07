import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sort import RandomSelection

def SelectionSortGenerator(a: np.ndarray, inplace: bool = True):
    '''
    Exact copy of SelectionSort function from sort.py
    but with a yield statement after each iteration.
    This makes the function a generator 
    and allows to animate the sorting process
    '''
    if inplace:
        ar = a
    else:
        ar = np.copy(a)

    for i in range(len(ar)-1):
        i_min = i
        for j in range(i+1,len(ar)):
            if ar[j]<ar[i_min]:
                i_min = j
        if i_min != i:
            ar[i] , ar[i_min] = ar[i_min] , ar[i]
        yield np.copy(ar)

def animate_sort(size: int = 100, fps: int = 10, dpi: int = 300, name: str = 'sorting_animation'):
    '''
    Function to animate the selection sort algorithm
    The process is the exact same as in sort.py
    but with an additional step of creating an animation
    '''
    samp = np.load('output/sample.npy')
    samp = samp[0]
    rand_samp = RandomSelection(samp, N=size, seed=2)

    fig, ax = plt.subplots()
    bars = ax.bar(np.arange(len(rand_samp)), rand_samp, color='crimson')
    ax.set_title('Selection Sort Animation')
    ax.set_ylabel('Value')
    ax.set_xlabel('Index')

    # Sorting steps generator
    sort_steps = list(SelectionSortGenerator(rand_samp))

    def update(frame):
        # Update the heights of the bars
        for bar, height in zip(bars, frame):
            bar.set_height(height)

        return bars

    # Create the animation
    ani = FuncAnimation(fig, update, frames=sort_steps, blit=True)

    # Save the animation as a GIF
    gif_writer = PillowWriter(fps=fps)
    ani.save(f'plots/{name}.gif', writer=gif_writer, dpi=300)

if __name__ == '__main__':
    animate_sort()
    animate_sort(size=20, fps=5, dpi=300, name='sorting_animation_small')
    animate_sort(size=200, fps=20, dpi=300, name='sorting_animation_large')
