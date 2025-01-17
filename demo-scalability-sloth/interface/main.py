import os
import sys

from PySide6.QtWidgets import (
    QApplication, 
    QWidget, 
    QLineEdit,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QTableWidget, QTableWidgetItem, 
    QPushButton, 
    QListWidget, 
    QStackedWidget,
    QFileDialog
)

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QIntValidator

from dltf.sloth.sloth import sloth
from dltf.testers.josie.josie import JOSIETester
from dltf.utils.datalake import MongoDBDataLakeHandler


query_ids = [
    38,
    169633,
    520835,
    558808,
    572002,
    939555,
    9892,
    1003853,
    1037572,
    751207
]



class TableComparisonApp(QWidget):
    def __init__(self, searcher:JOSIETester):
        super().__init__()
        self.searcher = searcher

        # Initialize the UI
        self.setWindowTitle("SLOTH Scalability Demo")
        self.setGeometry(200, 200, 1680, 1050)
        
        # Create the layout
        main_layout = QHBoxLayout()  # Horizontal layout for left, center, and right parts

        ########################### LEFT ###########################
        
        self.open_file_button = QPushButton('Load CSV table')
        self.open_file_button.clicked.connect(self.open_file_dialog)
        
        # Left table (Main table)
        self.query_table_w = QTableWidget(self)
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.open_file_button)
        left_layout.addWidget(self.query_table_w)
        
        # Center stacked widget for showing the selected table
        self.center_widget = QStackedWidget(self)

        # Add the initial "empty" table to the center widget
        self.right_table_w = QTableWidget(self)
        self.center_widget.addWidget(self.right_table_w)

        # Populate the main table (left table) with data
        self.query_table = None
        right_layout_width = 250

        # Right list of IDs with a scrollbar
        self.id_list = QListWidget(self)
        self.id_list.setFixedWidth(right_layout_width)
        # self.id_list.addItems(["ID 1", "ID 2", "ID 3", "ID 4", "ID 5"])  # Example IDs

        # Create the text to enter the number of results K
        self.k_lineedit = QLineEdit()
        self.min_h_lineedit = QLineEdit()
        self.min_w_lineedit = QLineEdit()
        
        self.k_lineedit.setValidator(QIntValidator())
        self.min_h_lineedit.setValidator(QIntValidator())
        self.min_w_lineedit.setValidator(QIntValidator())

        self.k_lineedit.setFixedWidth(right_layout_width)
        self.min_h_lineedit.setFixedWidth(right_layout_width)
        self.min_w_lineedit.setFixedWidth(right_layout_width)

        parameters_layout = QFormLayout()
        parameters_layout.addRow("Number of results:", self.k_lineedit)
        parameters_layout.addRow("Minimum number of rows:", self.min_h_lineedit)
        parameters_layout.addRow("Minimum number of columns:", self.min_w_lineedit)

        # Create JOSIE search button
        self.josie_search_button = QPushButton("Search candidates with JOSIE", self)
        self.josie_search_button.clicked.connect(self.josie_search)
        self.josie_search_button.setFixedWidth(right_layout_width)

        # Create Compare button
        self.compare_button = QPushButton("Compare Tables", self)
        self.compare_button.clicked.connect(self.compare_tables)
        self.compare_button.setFixedWidth(right_layout_width)

        # Create Clear button
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setFixedWidth(right_layout_width)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.id_list)
        right_layout.addLayout(parameters_layout)
        right_layout.addWidget(self.josie_search_button, alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.compare_button, alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.clear_button, alignment=Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignBottom)

        # Set up the layout
        main_layout.addLayout(left_layout)
        # main_layout.addWidget(self.query_table_w)
        main_layout.addWidget(self.center_widget)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

        # Connect the list item click to load data into the center table
        self.id_list.itemClicked.connect(self.load_data_for_id)

    def open_file_dialog(self):
        print('2', os.environ['XDG_DATA_DIRS'])
        dialog = QFileDialog(self)

        dialog.setDirectory(f'{os.path.dirname(__file__)}')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("CSV (*.csv)")
        # dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                pass


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

    def josie_search(self):
        if self.query_table == None:
            return
        

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
    data_path                   = f'{os.path.dirname(__file__)}/../data'
    tmp_path                    = f'{os.path.dirname(__file__)}/../tmp'

    # Set up the DataLake handler
    datalake_name               = 'demo'
    datalake_location           = 'mongodb'
    datasets                    = ['sloth.demo']
    dlh                         = MongoDBDataLakeHandler(datalake_location, datalake_name, datasets)

    # create data folder if it doesn't exist
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # JOSIE (global search tool) parameters
    mode                        = 'bag'
    blacklist                   = set() # set(['–', '—', '-', '●', '&nbsp', '&nbsp;', '&nbsp; &nbsp;', 'yes', 'no', 'n/a', 'none', '{{y}}', '{{n}}', '{{yes}}', '{{no}}', '{{n/a}}'] + list(map(str, range(1000))))
    string_translators          = ['whitespace', 'lowercase']
    string_patterns             = []
    force_sampling_cost         = False # force JOSIE to do cost sampling before querying
    token_table_on_memory       = False # build the token table used by JOSIE directly on disk
    tokens_bidict_file          = f'{data_path}/tokens-bidict.pickle'
    results_file                = f'{tmp_path}/tmpresults.csv'

    # connection info for the JOSIE inverted index
    db_config = {
        'drivername': 'postgresql',
        'database'  : 'DEMODB',
        'port'      :  5442,
        'host'      : 'localhost',
        'username'  : 'demo',
        'password'  : 'demo',
    }

    # Instatiate JOSIE
    josie = JOSIETester(
        mode=mode,
        blacklist=blacklist,
        datalake_handler=dlh,
        string_translators=string_translators,
        string_patterns=string_patterns,
        dbstatfile=None,
        tokens_bidict_file=tokens_bidict_file,
        josie_db_connection_info=db_config,
        spark_config=None
    )

    app = QApplication(sys.argv)
    # os.environ['QT_DEBUG_PLUGINS'] = '1'
    # print(os.environ['QT_DEBUG_PLUGINS'])
    # print('1', os.environ['XDG_DATA_DIRS'])
    window = TableComparisonApp(josie)
    window.show()
    sys.exit(app.exec())
