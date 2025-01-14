import pymongo
import tabulate


mongoclient = pymongo.MongoClient()
collection = mongoclient.sloth.latest_snapshot_tables

search_tokens = {' city'}
filter_tokens = {'champ', 'disc', 'race', 'olymp', 'result', 'minist', 'member', 'list', 'york', 'kansas', 'toronto', 'junction', 'minnesota'}
start_from = 0


for i, doc in enumerate(collection.find({'_id_numeric': {'$gt': start_from}})):
    if any(token in str(doc['context']).lower() for token in search_tokens) and not any(token in str(doc['context']).lower() for token in filter_tokens):
        print(f'Document {i} - {doc["_id"]} - {doc["_id_numeric"]} - {doc["context"]}')
        r = input('display? (y/n/exit) ')
        if r in {'n', 'no'}:
            continue
        elif r == 'exit':
            print(r)
            break
        else:
            headers = doc['content'][:doc['num_header_rows']][0] if doc['num_header_rows'] > 0 else []
            print(tabulate.tabulate(doc['content'][doc['num_header_rows']:],
                                    headers=headers, tablefmt='simple_outline'))
            print()


# FOOTBALL
# Document 37 - 618.80527 - 38 - ['Paul Gallagher (footballer)', 'Career statistics | Club']


# POLLUT
# Document 182372 - 141.81431 - 182372 - ['Cumberland Fossil Plant', 'Pollution and releases into environment']

# APARTMENT

# PATHO
# Document 31713 - 94.29391 - 31713 - ['Waterborne diseases', 'Infections by type of pathogen | Protozoa']
# Document 43091 - 617.67363 - 43091 - ['Bartonella', 'Pathophysiology']
# Document 74170 - 591.41857 - 74170 - ['Erythromelalgia', 'Pathophysiology']
# Document 79178 - 623.4541 - 79178 - ['Pathognomonic', 'Examples']
# Document 111351 - 88.34343 - 111351 - ['Bartter syndrome', 'Pathophysiology']
# Document 123172 - 66.45217 - 123172 - ['Innate immune system', 'Pathogen-specificity']
# Document 149698 - 226.5462 - 149698 - ['Pathogenic bacteria', 'List of genera and microscopy features']
# Document 183796 - 587.69413 - 183796 - ['Hypopituitarism', 'Pathophysiology']
# Document 242348 - 305.87654 - 242348 - ['Timeline of myocardial infarction pathology', '']
# Document 267433 - 312.105582 - 267433 - ['Keratin disease', 'Pathology']
# Document 521564 - 171.81789 - 521564 - ['Meckel syndrome', 'Pathophysiology']
# Document 526442 - 201.90071 - 526442 - ['Copper deficiency', 'Pathophysiology']
# Document 527362 - 567.89412 - 527362 - ['Hereditary spherocytosis', 'Pathophysiology']
# Document 548843 - 369.34100 - 548843 - ['Experiential avoidance', 'Relevance to psychopathology']
# Document 558801 - 57.67864 - 558801 - ['Alphavirus', 'Pathogenesis and immune response']
# Document 897485 - 597.50535 - 897485 - ['Siderophore', 'Ecology | Animal pathogens']
# Document 1086615 - 27.27959 - 1086615 - ['Streptococcus', 'Pathogenesis and classification']
# Document 1157584 - 558.42446 - 1157584 - ['Ovarian cancer', 'Pathophysiology']
# Document 1262726 - 292.48233 - 1262726 - ['International Pathogenic Neisseria Conference', 'List of Conferencesname:0']


# CONSUME
# Document 670815 - 370.29317 - 670815 - ['Consumer Financial Protection Bureau', 'List of directors']
# Document 684728 - 173.2500 - 684728 - ['Consumerist', 'Features']
# Document 928317 - 119.13078 - 928317 - ['List of longest consumer road vehicles', 'Longest vehicles | Longest sedans']
# Document 1356495 - 650.85176 - 1356495 - ['Consumers Energy 400', 'Past winners']
# Document 1523569 - 533.9250 - 1523569 - ['Market segmentation', 'Bases for segmenting consumer markets | Other types of consumer segmentation | Generational segments']

# CANCER
# Document 12201 - 603.68439 - 12201 - ['Hereditary nonpolyposis colorectal cancer', 'Genetics']
# Document 61686 - 200.9898 - 61686 - ['Veterinary oncology', 'Cancer statistics | Male dogs']
# Document 141534 - 235.50870 - 141534 - ['Hypopharyngeal cancer', 'Diagnosis | Stages and survival rates | Early stage']
# Document 260919 - 311.9043 - 260919 - ['Epidemiology of cancer', 'Rates and mortality | United States']
# Document 272974 - 558.42455 - 272974 - ['Ovarian cancer', 'Diagnosis | Pathology']
# Document 306109 - 70.23808 - 306109 - ['Prostate cancer staging', 'TNM staging | Overall staging']
# Document 653831 - 565.11654 - 653831 - ['Gene duplication', 'As amplification | Role in cancer']


# CIT
# Document 2319 - 575.51108 - ['Citroën XM', 'Engines']

# Document 2976 - 167.25485 - ['List of citrus diseases', 'Bacterial diseases']
# Document 3653 - 167.25481 - ['List of citrus diseases', 'Fungal diseases']
# Document 3654 - 167.25456 - ['List of citrus diseases', 'Nematodes, parasitic']
# Document 3655 - 167.25472 - ['List of citrus diseases', 'Viroids and graft-transmissible pathogens [GTP]']

# Document 3871 - 601.59871 - ['B (New York City Subway service)', 'Route | Service pattern']
# Document 3872 - 601.59874 - ['B (New York City Subway service)', 'Route | Stations']

# Document 4182 - 118.102263 - ['Prince Consort-class ironclad', 'Carrying capacity of wood hulls compared with iron hulls']


# ENV
# Document 8041 - 170.72724 - ['Laminopathy', 'Diagnosis | Types of known laminopathies and other nuclear envelopathies']
# Document 10162 - 556.83430 - ['Gold cyanidation', 'Effects on the environment']
# Document 10706 - 187.68156 - ['Comparison of audio synthesis environments', 'General']
# Document 11467 - 187.16302 - ['Green chemistry metrics', 'Environmental (E) factor']


# CIT, ENV
# Document 102531 - 578.84187 - ['Environmental impact of electricity generation', 'Water usage']
# 
# Document 106147 - 647.20234 - ['Citrus canker', 'Favorable environmental condition | Susceptibility']
# 
# Document 184040 - 203.73738 - ['List of tallest buildings in Kansas City, Missouri', 'Buildings proposed / under construction / envisioned | Envisioned']
# 
# Document 279885 - 156.16487 - ['New York City Department of Environmental Protection', 'Facilities | Wastewater treatment']
# Document 880386 - 36.8303 - ['Electricity generation', 'Environmental concerns']


# COUNTRY
# Document 4190 - 613.14617 - ['List of association football stadiums by country', 'Belgium']
# Document 8635 - 170.64318 - ['Restrictions on the import of cryptography', 'Status by country']
# 
# Document 18252 - 603.20489 - ['List of FIS Cross-Country World Cup champions', 'Men | Overall']
#
# Document 24144 - 176.65599 - ['List of South Asian television channels by country', 'List of channels | United States']
#
# Document 26680 - 177.30502 - ['Canadian Country Music Association', 'CCMA Awards | Awards by year']


# POPULATION
# Document 4958 - 29.31705 - ['Bavaria', 'Administrative divisions | Administrative districts | Population and area']
# Document 7561 - 625.56275 - ['Fraser Valley Regional District', 'Population']

# SPACE
# Document 2597 - 517.38896 - ['Space Telescope Science Institute', 'STScI activities | Science instrument calibration and characterization']
# Document 8607 - 170.69219 - ['2008 in spaceflight', 'Orbital launch statistics | By rocket | By configuration']
# Document 12833 - 608.12370 - ['Delta-v budget', 'Budget | anchorEarth-Moon space budgetEarth–Moon space budgetEarth-MoonEarth–Moon Earth–Moon space—high thrust']
