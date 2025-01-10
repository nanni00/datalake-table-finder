import sys

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QListWidget, QStackedWidget

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush

from dltf.sloth.sloth import sloth


class TableComparisonApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the UI
        self.setWindowTitle("Demo Interface")
        self.setGeometry(100, 100, 1000, 400)
        
        # Create the layout
        main_layout = QHBoxLayout()  # Horizontal layout for left, center, and right parts

        # Left table (Main table)
        self.query_table_w = QTableWidget(self)
        self.query_table_w.setFixedWidth(300)

        # Center stacked widget for showing the selected table
        self.center_widget = QStackedWidget(self)

        # Add the initial "empty" table to the center widget
        self.right_table_w = QTableWidget(self)
        self.center_widget.addWidget(self.right_table_w)

        # Populate the main table (left table) with data
        self.query_table = [
            [1, "Alice", "Engineer"],
            [2, "Bob", "Designer"],
            [3, "Charlie", "Manager"]
        ]
        self.load_table(self.query_table_w, self.query_table)

        # Right list of IDs with a scrollbar
        self.id_list = QListWidget(self)
        self.id_list.setFixedWidth(150)
        self.id_list.addItems(["ID 1", "ID 2", "ID 3", "ID 4", "ID 5"])  # Example IDs

        # Create Compare button
        self.compare_button = QPushButton("Compare Tables", self)
        self.compare_button.clicked.connect(self.compare_tables)

        # Create Clear button
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.id_list)
        right_layout.addWidget(self.compare_button, alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.clear_button, alignment=Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignBottom)

        # Set up the layout
        main_layout.addWidget(self.query_table_w)
        main_layout.addWidget(self.center_widget)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

        # Connect the list item click to load data into the center table
        self.id_list.itemClicked.connect(self.load_data_for_id)

    def load_table(self, table, data):
        """Populate a QTableWidget with data"""
        table.setRowCount(len(data))
        table.setColumnCount(len(data[0]))

        for i, row in enumerate(data):
            for j, item in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(item)))

    def load_data_for_id(self, item):
        """Load new data in the center table based on the selected ID"""
        id_clicked = item.text()

        rows1 = self.query_table_w.rowCount()
        cols1 = self.query_table_w.columnCount()

        # Clear previous comparisons
        for row in range(rows1):
            for col in range(cols1):
                item = self.query_table_w.item(row, col)
                if item:
                    item.setBackground(QBrush())

        # Example logic: Select data based on ID
        if id_clicked == "ID 1":
            data = [
                [1, "Alice", "Engineer"],
                [2, "Bob", "Designer"]
            ]
        elif id_clicked == "ID 2":
            data = [
                [3, "Charlie", "Manager"],
                [4, "David", "Architect"]
            ]
        elif id_clicked == "ID 3":
            data = [
                [5, "Eve", "Consultant"],
                [6, "Frank", "Scientist"]
            ]
        elif id_clicked == "ID 4":
            data = [
                [7, "Grace", "Doctor"],
                [8, "Hank", "Artist"]
            ]
        elif id_clicked == "ID 5":
            data = [
                [9, "Ivy", "Teacher"],
                [10, "Jack", "Writer"]
            ]

        # Clear previous data and load new data into center table
        self.right_table_w.clear()
        self.load_table(self.right_table_w, data)
        self.right_table = data

        # Switch to the updated table view in the center
        self.center_widget.setCurrentWidget(self.right_table_w)

    def clear(self):
        rows1 = self.query_table_w.rowCount()
        rows2 = self.right_table_w.rowCount()
        cols1 = self.query_table_w.columnCount()
        cols2 = self.right_table_w.columnCount()

        # Clear previous comparisons
        for row in range(rows1):
            for col in range(cols1):
                item = self.query_table_w.item(row, col)
                if item:
                    item.setBackground(QBrush())

        for row in range(rows2):
            for col in range(cols2):
                item = self.right_table_w.item(row, col)
                if item:
                    item.setBackground(QBrush())

    def compare_tables(self):
        """Compare tables and highlight largest overlap"""
        qrows = self.query_table_w.rowCount()
        rrows = self.right_table_w.rowCount()
        qcols = self.query_table_w.columnCount()
        rcols = self.right_table_w.columnCount()

        metrics = []
        results, metrics = sloth(self.query_table, self.right_table, metrics=metrics, verbose=False, complete=False)
        # Clear previous comparisons
        for row in range(qrows):
            for col in range(qcols):
                item = self.query_table_w.item(row, col)
                if item:
                    item.setBackground(QBrush())

        for row in range(rrows):
            for col in range(rcols):
                item = self.right_table_w.item(row, col)
                if item:
                    item.setBackground(QBrush())

        if metrics[0] != 0:
            commons = list(zip(*(map(str, res) for res in results[0][1])))
            matches = list(zip(results[0][0], commons))

            for (qrow, rrow), values in matches:
                for col in range(qcols):
                    item = self.query_table_w.item(qrow, col)
                    if item and item.data(0) in values:
                        item.setBackground(Qt.GlobalColor.darkGreen)
                    
                for col in range(rcols):
                    item = self.right_table_w.item(rrow, col)
                    if item and item.data(0) in values:
                        item.setBackground(Qt.GlobalColor.darkGreen)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TableComparisonApp()
    window.show()
    sys.exit(app.exec())
