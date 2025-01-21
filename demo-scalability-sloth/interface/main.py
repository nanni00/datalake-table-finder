import os
import pickle
import re
import sys

import polars as pl

from PySide6.QtWidgets import (
    QApplication, 
    QWidget, 
    QLineEdit,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QTableWidget, QTableWidgetItem, 
    QPushButton, 
    QListWidget,
    QDialogButtonBox, QFileDialog,
    QCheckBox,
    QToolBar,
    QMenu,
    QMainWindow,
    QLabel
)

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QIntValidator, QAction, QCursor

from dltf.sloth.sloth import sloth
from dltf.utils.misc import clean_string
from dltf.utils import tables
from dltf.gsa.josie.josie import JOSIEGS
from dltf.utils.datalake import MongoDBDataLakeHandler



def prepare_query(qdoc, tokens_bidict, string_blacklist, string_translators, string_patterns, mode):    
    # Extract a bag of tokens from the document's content
    query_set = tables.table_to_tokens(
        table=qdoc['content'], 
        valid_columns=qdoc['valid_columns'], 
        mode=mode, 
        string_blacklist=string_blacklist,
        string_translators=string_translators,
        string_patterns=string_patterns
    )
    
    # Map each token in the sets with its correspondent token ID for JOSIE
    query_set = sorted(
        [
            tokens_bidict.inverse[clean_string(token, string_translators, string_patterns)]
            for token in query_set 
            if clean_string(token, string_translators, string_patterns) in tokens_bidict.inverse
        ]
    )

    return {qdoc['_id_numeric']: query_set}


def get_result_ids(s):
    return list(map(int, re.findall(r'\d+', s)[::2]))

def get_result_overlaps(s):
    return list(map(int, re.findall(r'\d+', s)[1::2]))




class MyTable(QTableWidget):
    pass


class DemoMainWindow(QMainWindow):
    def __init__(self, searcher:JOSIEGS):
        super().__init__()
        self.searcher = searcher
        self.dlh = searcher.dlh
        with open(self.searcher.tokens_bidict_file, 'rb') as fr:
            self.tokens_bidict = pickle.load(fr)
        self.string_blacklist = set()
        self.string_translators = []
        self.string_patterns = []
        self.table_translation_mode = 'bag'

        # Initialize the UI
        self.setWindowTitle("SLOTH Scalability Demo")
        self.setGeometry(200, 200, 1680, 1050)
        

        ########################### LEFT ###########################
        
        self.open_file_button = QPushButton('Load CSV table')
        self.open_file_button.clicked.connect(self.open_file_dialog)

        # Populate the main table (left table) with data
        self.query_table = None
        
        # Left table (Main table)
        self.query_table_w = MyTable(self)
        self.query_table_w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.query_table_w.customContextMenuRequested.connect(self.emptySpaceMenu)
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.query_table_w)
        

        ########################## CENTER ##########################
        
        # Add the initial "empty" table to the center widget
        self.result_table_obj = []
        self.result_table_w = MyTable(self)
        self.query_table_w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.query_table_w.customContextMenuRequested.connect(self.emptySpaceMenu)
        
        # Center stacked widget for showing the selected table
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.result_table_w)
        
        
        ########################### RIGHT ###########################
        # Right list of IDs with a scrollbar
        self.id_list = QListWidget(self)
        self.id_list.itemClicked.connect(self.load_data_for_id)

        translators_linetext = QLabel('Translators:')
        self.lowercase_cb   = QCheckBox('Lowercase')
        self.whitespace_cb  = QCheckBox('Whitespace')
        self.punctuation_cb = QCheckBox('Punctuation')
        
        # Create form for search parameters
        self.k_lineedit = QLineEdit()
        self.min_h_lineedit = QLineEdit()
        self.min_w_lineedit = QLineEdit()
        
        # Set a Integer Validator for each line of the form
        self.k_lineedit.setValidator(QIntValidator(1, int(1e9)))
        self.min_h_lineedit.setValidator(QIntValidator(0, int(1e9)))
        self.min_w_lineedit.setValidator(QIntValidator(0, int(1e9)))

        # Define some default values
        self.k_lineedit.setText('5')
        self.min_h_lineedit.setText('0')
        self.min_w_lineedit.setText('0')

        parameters_layout = QFormLayout()
        parameters_layout.addRow("Number of results:", self.k_lineedit)
        parameters_layout.addRow("Minimum number of rows:", self.min_h_lineedit)
        parameters_layout.addRow("Minimum number of columns:", self.min_w_lineedit)

        # Create JOSIE search button
        self.josie_search_button = QPushButton("Search candidates with JOSIE", self)
        self.josie_search_button.clicked.connect(self.search)

        # Create Compare button
        self.compare_button = QPushButton("Compare Tables with SLOTH", self)
        self.compare_button.clicked.connect(self.compare_tables)

        # Create Clear button
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.id_list)
        right_layout.addWidget(translators_linetext)
        right_layout.addWidget(self.lowercase_cb)
        right_layout.addWidget(self.whitespace_cb)
        right_layout.addWidget(self.punctuation_cb)
        right_layout.addLayout(parameters_layout)
        right_layout.addWidget(self.josie_search_button)
        right_layout.addWidget(self.compare_button)
        right_layout.addWidget(self.clear_button)
        
        self.create_toolbar()

        ########################## FINAL SETUP #########################        

        # Create the layout
        central_widget = QWidget()
        
        # Horizontal layout for left, center, and right parts
        main_layout = QHBoxLayout()  

        # Set up the final layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(center_layout)
        main_layout.addLayout(right_layout)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_toolbar(self):
        toolbar = QToolBar("The ToolBar")
        self.addToolBar(toolbar)
        
        load_csv_action     = QAction('Load CSV', self)
        # blacklist_action    = QAction('Blacklist', self)
        # translator_action   = QAction('Translators', self)
        # pattern_action      = QAction('Patterns', self)

        load_csv_action.triggered.connect(self.open_file_dialog)
        # translator_action.triggered.connect(self.open_translators_menu)
        
        toolbar.addAction(load_csv_action)
        # toolbar.addAction(blacklist_action)
        # toolbar.addAction(translator_action)
        # toolbar.addAction(pattern_action)


    def emptySpaceMenu(self, event):
        menu = QMenu()
        index = self.query_table_w.indexAt(event)
        add_to_blacklist_action = QAction('')
        remove_from_blacklist_action = QAction('')

        if index.isValid():
            add_to_blacklist_action.setText(f'Add token "{index.data(0)}" to blacklist')
            remove_from_blacklist_action.setText(f'Remove token "{index.data(0)}" from blacklist')
        
        menu.addAction(add_to_blacklist_action)
        menu.addAction(remove_from_blacklist_action)

        res = menu.exec(QCursor.pos())

        if res == add_to_blacklist_action:
            self.string_blacklist.add(index.data(0))
        elif res == remove_from_blacklist_action and index.data(0) in self.string_blacklist:
            self.string_blacklist.remove(index.data(0))


    def open_file_dialog(self):
        dialog = QFileDialog(self)

        dialog.setDirectory(f'{os.path.dirname(__file__)}')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("CSV (*.csv)")
        # dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                data = pl.read_csv(filenames[0]).rows()
                self.query_table = data
                self.set_table_on_widget(self.query_table_w, data)

    def open_translators_menu(self):
        menu = QMenu(self)
        option1 = QAction("Option 1", self, checkable=True)
        option1.setChecked(True)  # Set the default state of the option
        option2 = QAction("Option 2", self, checkable=True)
        option3 = QAction("Option 3", self, checkable=True)

        # Connect the actions to a slot that updates their states
        option1.triggered.connect(lambda: self.option_toggled(option1))
        option2.triggered.connect(lambda: self.option_toggled(option2))
        option3.triggered.connect(lambda: self.option_toggled(option3))

        # Add the actions to the menu
        menu.addAction(option1)
        menu.addAction(option2)
        menu.addAction(option3)

    def set_table_on_widget(self, table_w, table_data):
        """Populate a QTableWidget with table data (as a list-of-rows)"""
        table_w.clear()
        table_w.setRowCount(len(table_data))
        table_w.setColumnCount(len(table_data[0]))

        for i, row in enumerate(table_data):
            for j, item in enumerate(row):
                table_w.setItem(i, j, QTableWidgetItem(str(item)))

    def load_data_for_id(self, item):
        """Load new data in the center table based on the selected ID"""
        id_clicked = int(item.text())

        rows1 = self.query_table_w.rowCount()
        cols1 = self.query_table_w.columnCount()

        # Clear previous comparisons
        for row in range(rows1):
            for col in range(cols1):
                item = self.query_table_w.item(row, col)
                if item:
                    item.setBackground(QBrush())
        
        print(f'{id_clicked=}')
        self.result_table_obj = self.dlh.get_table_by_numeric_id(id_clicked)
        table_data = self.result_table_obj['content']

        # Clear previous data and load new data into center table
        self.result_table_w.clear()
        self.set_table_on_widget(self.result_table_w, table_data)

    def search(self):
        if self.query_table == None:
            print('Load query table before searching')
            return
        if not (self.k_lineedit.hasAcceptableInput() and self.min_h_lineedit.hasAcceptableInput() and self.min_w_lineedit.hasAcceptableInput()):
            print('Parameters K, min_w and min_h must be in range (1, inf), (0, inf), (0, inf)')
            return

        k = int(self.k_lineedit.text())
        min_h = int(self.min_h_lineedit.text())
        min_w = int(self.min_w_lineedit.text())

        # Get CheckBoxes statuses
        translators = []
        if self.lowercase_cb.checkState().value == Qt.CheckState.Checked:
            translators.append('lowecase')
        if self.whitespace_cb.checkState().value == Qt.CheckState.Checked:
            translators.append('whitespace')
        if self.lowercase_cb.checkState().value == Qt.CheckState.Checked:
            translators.append('punctuation')

        print(f'{self.string_blacklist=}')


        qdoc = {'_id_numeric': -1, 'content': self.query_table, 'valid_columns': [1] * len(self.query_table[0])}
        _, results = self.searcher.query(
            None, 
            k, 
            prepare_query(
                qdoc, self.tokens_bidict, 
                self.string_blacklist, translators, self.string_patterns, 
                self.table_translation_mode
                )
            )
        self.id_list.clear()
        self.id_list.addItems([str(i) for i, o in results[0][1]])
        
    def clear(self):
        self.query_table_w.clear()
        self.result_table_w.clear()

    def compare_tables(self):
        """Compare tables and highlight largest overlap"""
        qrows = self.query_table_w.rowCount()
        rrows = self.result_table_w.rowCount()
        qcols = self.query_table_w.columnCount()
        rcols = self.result_table_w.columnCount()

        metrics = []
        results, metrics = sloth(self.query_table, self.result_table_obj['content'], metrics=metrics, verbose=False, complete=False)
        
        # Clear previous comparisons
        for row in range(qrows):
            for col in range(qcols):
                item = self.query_table_w.item(row, col)
                if item:
                    item.setBackground(QBrush())

        for row in range(rrows):
            for col in range(rcols):
                item = self.result_table_w.item(row, col)
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
                    item = self.result_table_w.item(rrow, col)
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
    josie = JOSIEGS(
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
    # app.setPalette(QPalette(QColor("white")))
    window = DemoMainWindow(josie)
    window.show()
    sys.exit(app.exec())
