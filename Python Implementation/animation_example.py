#import numpy as np
#import numpy.random as npr
#import matplotlib.pyplot as plt
#from matplotlib import animation
#
## https://stackoverflow.com/questions/22010586/matplotlib-animation-duration
#
#
#n = 10
#A0 = npr.randn(n,n)
#fig,ax=plt.subplots()
#im=plt.imshow(A0,cmap='afmhot_r',animated=True)
#def update(t):
#    global A
#    A = A0*np.cos(0.01*2*np.pi*t)
#    im.set_data(A)
#    return im,
#a=animation.FuncAnimation(fig,update,interval=10,blit=True)
#
## Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#a.save('animation.mp4', writer=writer)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15.
    y += np.pi / 20.
    im = plt.imshow(f(x, y), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

ani.save('dynamic_images.mp4')

plt.show()

#"""
#===================
#Saving an animation
#===================
#
#This example showcases the same animations as `basic_example.py`, but instead
#of displaying the animation to the user, it writes to files using a
#MovieWriter instance.
#"""
#
## -*- noplot -*-
#import numpy as np
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#Writer = animation.FFMpegWriter(fps=30, codec='libx264') # Or 
#
#
#def update_line(num, data, line):
#    line.set_data(data[..., :num])
#    return line,
#
## Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#
#
#fig1 = plt.figure()
#
#data = np.random.rand(2, 25)
#l, = plt.plot([], [], 'r-')
#plt.xlim(0, 1)
#plt.ylim(0, 1)
#plt.xlabel('x')
#plt.title('test')
#line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
#                                   interval=50, blit=True)
#line_ani.save('lines.mp4', writer=writer)
#
#fig2 = plt.figure()
#
#x = np.arange(-9, 10)
#y = np.arange(-9, 10).reshape(-1, 1)
#base = np.hypot(x, y)
#ims = []
#for add in np.arange(15):
#    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))
#
#im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
#                                   blit=True)
#im_ani.save('im.mp4', writer=Writer)