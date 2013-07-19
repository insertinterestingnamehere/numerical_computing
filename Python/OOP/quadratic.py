import sys
import cmath
from PyQt4 import QtGui, QtCore


class Quadratic(QtGui.QWidget):
	def __init__(self, val_step=.1):
		super(Quadratic, self).__init__()

		self.a = 0
		self.b = 0
		self.c = 0
		self.val_step = val_step
		self.calcRoots()

		self.initUI()

	def initUI(self):

		val_a = QtGui.QDoubleSpinBox(self)
		val_a.valueChanged.connect(self.updatea)
		val_b = QtGui.QDoubleSpinBox(self)
		val_b.valueChanged.connect(self.updateb)
		val_c = QtGui.QDoubleSpinBox(self)
		val_c.valueChanged.connect(self.updatec)

		for x in (val_a, val_b, val_c):
			x.setSingleStep(self.val_step)
			x.setMinimum(-20)
			x.setMaximum(20)

		dispgrid = QtGui.QGridLayout()
		pos = QtGui.QLabel("Positive Root: ", self)
		neg = QtGui.QLabel("Negative Root: ", self)

		self.root_neg = QtGui.QLabel(self)
		self.root_pos = QtGui.QLabel(self)
		dispgrid.addWidget(neg, 0, 0)
		dispgrid.addWidget(pos, 1, 0)
		dispgrid.addWidget(self.root_neg, 0, 1)
		dispgrid.addWidget(self.root_pos, 1, 1)

		vbox = QtGui.QVBoxLayout()
		vbox.addWidget(val_a)
		vbox.addWidget(val_b)
		vbox.addWidget(val_c)
		vbox.addLayout(dispgrid)

		self.setLayout(vbox)
		self.setGeometry(100, 100, 300, 300)
		self.show()

	def updatea(self, a):
		self.a = a
		self.updateRoots()

	def updateb(self, b):
		self.b = b
		self.updateRoots()

	def updatec(self, c):
		self.c = c
		self.updateRoots()

	def calcRoots(self):
		a, b, c = self.a, self.b, self.c
		denom = 2.*a
		
		try:
			s = cmath.sqrt(b**2 - 4.0*a*c)
			self.rneg, self.rpos = ((-b - s)/(2*a),
						  			(-b + s)/(2*a))
		except ZeroDivisionError:
			self.rneg, self.rpos = 'err', 'err'

	def updateRoots(self):
		self.calcRoots()
		n, p = map(QtCore.QString, map(str, [self.rneg, self.rpos]))
		
		self.root_neg.setText(n)
		self.root_pos.setText(p)
		
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Quadratic()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()