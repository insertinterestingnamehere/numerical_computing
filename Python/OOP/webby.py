import sys
from PyQt4 import QtGui, QtCore, QtWebKit

class WebNavigator(QtGui.QWidget):
	def __init__(self, home="about:blank"):
		super(WebNavigator, self).__init__()

		self.home = QtCore.QUrl(home)
		self._initUI()

	def _initUI(self):
		self.wkctl = QtWebKit.QWebView(self)
		self.wkctl.setUrl(self.home)
		self.wkctl.urlChanged.connect(self.changedURL)

		prog_bar = QtGui.QProgressBar(self)
		self.wkctl.loadProgress.connect(prog_bar.setValue)

		navbar = QtGui.QHBoxLayout()
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

		navbar.addWidget(back)
		navbar.addWidget(forward)
		navbar.addWidget(self.stop_reload)
		navbar.addWidget(self.addr_bar)
		navbar.addWidget(go_button)

		vbox = QtGui.QVBoxLayout()
		vbox.addLayout(navbar)
		vbox.addWidget(self.wkctl)
		vbox.addWidget(prog_bar)

		self.setLayout(vbox)
		self.setGeometry(0, 0, 800, 600)
		self.setWindowTitle("Simple WebKit Navigator")
		self.show()

	def changedURL(self, url):
		self.addr_bar.setText(url.toString())

	def goURL(self):
		new_addr = QtCore.QUrl.fromUserInput(self.addr_bar.text())
		self.wkctl.load(new_addr)

	def stop_action(self):
		self.stop_reload.setText("Stop")
		self.stop_reload.clicked.connect(self.wkctl.stop)

	def reload_action(self):
		self.stop_reload.setText("Reload")
		self.stop_reload.clicked.connect(self.wkctl.reload)


def main():
	app = QtGui.QApplication(sys.argv)
	web = WebNavigator()
	sys.exit(app.exec_())

if __name__ == "__main__":
	main()