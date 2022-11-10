# Import the required libraries.
from  PyQt5.QtWidgets  import *
from  matplotlib.backends.backend_qt5agg  import  FigureCanvas
from  matplotlib.figure  import  Figure

# Create a class.  
class  MplWidget ( QWidget ):
    
    # Create a constructor.
    def  __init__ (self, parent=None):
        
        # Call the constructor of the parent class. 
        QWidget.__init__ (self,  parent)
        
        # Create a canvas to plot the matplotlib figure.
        self.canvas = FigureCanvas(Figure(facecolor='#19232d'))
        
        # Create a layout.
        vertical_layout  =  QVBoxLayout()
        
        # Add the canvas into the layout. 
        vertical_layout.addWidget(self.canvas)
        
        # Add an Axes to the canvas.
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        
        # Set the layout.
        self.setLayout(vertical_layout)