import sys
from PySide import QtGui, QtCore, QtWebKit


class WebNavigator(QtGui.QWidget):

    def __init__(self, home="about:blank"):
        super(WebNavigator, self).__init__()

        self.home = QtCore.QUrl(home)
        self._initUI()

    def _initUI(self):
        r'''Initialize the interface components
        
        Each component is initialized follow a very similiar procedure
        of creating a reference, setting initial attributes, and then linking
        the component to our methods in our program.
        
        These methods that we link to are the methods that actually do the work
        when we interact with the component
        '''
        #Create the WebView component
        self.wkctl = QtWebKit.QWebView(self)
        #Set initial state
        self.wkctl.setUrl(self.home)
        #Connect to method of our class
        self.wkctl.urlChanged.connect(self.changedURL)

        #Create ProgressBar component
        prog_bar = QtGui.QProgressBar(self)
        #Connect the signals sent by the WebView to the ProgressBar
        self.wkctl.loadProgress.connect(prog_bar.setValue)

        #Create a horizontal box layout manager
        navbar = QtGui.QHBoxLayout()
        #Create the components that will belong to the HBox
        #Since we don't need to access the back and forward
        #outside of this method, we don't need to create class-wide references.
        back = QtGui.QPushButton("Back", self)
        back.clicked.connect(self.wkctl.back)
        forward = QtGui.QPushButton("Forward", self)
        forward.clicked.connect(self.wkctl.forward)
        self.stop_reload = QtGui.QPushButton("Reload", self)
        self.wkctl.loadStarted.connect(self.stop_action)
        self.wkctl.loadFinished.connect(self.reload_action)

        self.addr_bar = QtGui.QLineEdit(self)
        self.addr_bar.setText(self.home.toString())
        self.addr_bar.returnPressed.connect(self.goURL)

        go_button = QtGui.QPushButton("Go!", self)
        go_button.clicked.connect(self.goURL)

        #Add the widgets to the navbar
        navbar.addWidget(back)
        navbar.addWidget(forward)
        navbar.addWidget(self.stop_reload)
        navbar.addWidget(self.addr_bar)
        navbar.addWidget(go_button)

        #Now we create vertical Box layout
        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(navbar)
        vbox.addWidget(self.wkctl)
        vbox.addWidget(prog_bar)
        
        #Now assemble the entire window
        #Add the layout
        self.setLayout(vbox)
        #set the window size
        self.setGeometry(0, 0, 800, 600)
        #set the widow title
        self.setWindowTitle("Simple WebKit Navigator")
        #display the window
        self.show()

    def changedURL(self, url):
        '''The slot for the signal urlChanged of the WebView
        
        Everytime the url changes, we update the address bar with the new URL
        '''
        self.addr_bar.setText(url.toString())

    def goURL(self):
        '''Slot for clicked signal of the Go button.
        
        This will load the url in the address bar into the WebView.
        '''
        new_addr = QtCore.QUrl.fromUserInput(self.addr_bar.text())
        self.wkctl.load(new_addr)

    def stop_action(self):
        '''Slot for when the WebView is busy loading a URL and the Stop/Reload
        button is pushed.
        
        This will change the label of the button.
        It dynamically changes the connection between the button and the stop
        function.
        '''
        self.stop_reload.setText("Stop")
        self.stop_reload.clicked.connect(self.wkctl.stop)

    def reload_action(self):
        '''Very similar to the stop function, except it connects the button
        to the reload function.
        '''
        self.stop_reload.setText("Reload")
        self.stop_reload.clicked.connect(self.wkctl.reload)


def main():
    app = QtGui.QApplication(sys.argv)
    web = WebNavigator()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
