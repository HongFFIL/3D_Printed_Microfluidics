# main.py

import sys
from PyQt5.QtWidgets import QApplication
from gui.label_tool import LabelTool

def main():
    app = QApplication(sys.argv)
    window = LabelTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
