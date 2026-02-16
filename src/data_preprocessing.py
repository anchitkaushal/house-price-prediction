import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# LOAD DATA
def load_data(path):
    return pd.read_csv(path)



# HANDLE STRUCTURAL MISSING
def handle_structural_missing(data):

    basement_cols = ['BsmtQual','BsmtCond','BsmtExposure',
                     'BsmtFinType1','BsmtFinType2']
    
    data[basement_cols] = data[basement_cols].fillna('NoBasement')

    garage_cols = ['GarageQual','GarageCond',
                   'GarageType','GarageFinish']
    
    data[garage_cols] = data[garage_cols].fillna('NoGarage')
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)

    data['FireplaceQu'] = data['FireplaceQu'].fillna('NoFireplace')
    data['PoolQC'] = data['PoolQC'].fillna('NoPoolArea')
    data['Fence'] = data['Fence'].fillna('NoFence')
    data['MiscFeature'] = data['MiscFeature'].fillna('None')
    data['MasVnrType'] = data['MasVnrType'].fillna('None')
    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
    data['Alley'] = data['Alley'].fillna('NoAlley')

    return data



# HANDLE ACCIDENTAL MISSING
def handle_accidental_missing(data):

    data['Electrical'] = data['Electrical'].fillna(
        data['Electrical'].mode()[0]
    )

    data['LotFrontage'] = data['LotFrontage'].fillna(
        data['LotFrontage'].median()
    )

    return data

# ENCODING
def encode_data(data):

    ordinal_features = [
    'LotShape',
    'Utilities',
    'LandSlope',
    'ExterQual',
    'ExterCond',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'HeatingQC',
    'KitchenQual',
    'Functional',
    'FireplaceQu',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PavedDrive',
    'PoolQC',
    'Fence'
    ]
    ordinal_categories = [
            ['Reg', 'IR1', 'IR2', 'IR3'],                       # LotShape
    ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],               # Utilities
    ['Sev', 'Mod', 'Gtl'],                              # LandSlope
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],                     # ExterQual
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],                     # ExterCond
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],               # BsmtQual
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],               # BsmtCond
    ['NA', 'No', 'Mn', 'Av', 'Gd'],                     # BsmtExposure
    ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],   # BsmtFinType1
    ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],   # BsmtFinType2
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],                     # HeatingQC
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],                     # KitchenQual
    ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],  # Functional
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],               # FireplaceQu
    ['NA', 'Unf', 'RFn', 'Fin'],                        # GarageFinish
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],               # GarageQual
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],               # GarageCond
    ['N', 'P', 'Y'],                                    # PavedDrive
    ['NA', 'Fa', 'TA', 'Gd', 'Ex'],                     # PoolQC
    ['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']             # Fence
    ]

   

    

    nominal_features = [
    'MSZoning',
    'Street',
    'Alley',
    'LandContour',
    'LotConfig',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'Heating',
    'CentralAir',
    'Electrical',
    'GarageType',
    'MiscFeature',
    'SaleType',
    'SaleCondition'
    ]
    
    nominal_categories = [

    # MSZoning
    ['C (all)', 'FV', 'RH', 'RL', 'RM'],

    # Street
    ['Grvl', 'Pave'],

    # Alley
    ['Grvl', 'Pave', 'NoAlley'],

    # LandContour
    ['Lvl', 'Bnk', 'HLS', 'Low'],

    # LotConfig
    ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],

    # Neighborhood
    ['Blmngtn','Blueste','BrDale','BrkSide','ClearCr',
     'CollgCr','Crawfor','Edwards','Gilbert','IDOTRR',
     'MeadowV','Mitchel','NAmes','NoRidge','NPkVill',
     'NridgHt','NWAmes','OldTown','Sawyer','SawyerW',
     'Somerst','StoneBr','SWISU','Timber','Veenker'],

    # Condition1
    ['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe'],

    # Condition2
    ['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRAe'],

    # BldgType
    ['1Fam','2fmCon','Duplex','TwnhsE','Twnhs'],

    # HouseStyle
    ['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl'],

    # RoofStyle
    ['Flat','Gable','Gambrel','Hip','Mansard','Shed'],

    # RoofMatl
    ['ClyTile','CompShg','Membran','Metal','Roll',
     'Tar&Grv','WdShake','WdShngl'],

    # Exterior1st
    ['AsbShng','AsphShn','BrkComm','BrkFace','CBlock',
     'CemntBd','HdBoard','ImStucc','MetalSd','Plywood',
     'Stone','Stucco','VinylSd','Wd Sdng','WdShing'],

    # Exterior2nd
    ['AsbShng','AsphShn','Brk Cmn','BrkFace','CBlock',
     'CmentBd','HdBoard','ImStucc','MetalSd','Plywood',
     'Stone','Stucco','VinylSd','Wd Sdng','Wd Shng'],

    # MasVnrType
    ['BrkCmn','BrkFace','None','Stone'],

    # Foundation
    ['BrkTil','CBlock','PConc','Slab','Stone','Wood'],

    # Heating
    ['Floor','GasA','GasW','Grav','OthW','Wall'],

    # CentralAir
    ['N','Y'],

    # Electrical
    ['SBrkr','FuseA','FuseF','FuseP','Mix'],

    # GarageType
    ['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NoGarage'],

    # MiscFeature
    ['None','Gar2','Othr','Shed','TenC'],

    # SaleType
    ['WD','CWD','New','COD','Con','ConLw','ConLI','ConLD','Oth'],

    # SaleCondition
    ['Normal','Abnorml','AdjLand','Alloca','Family','Partial']
    ]
    
    ordinal_encoder = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    nominal_encoder = OneHotEncoder(
        categories=nominal_categories,
        handle_unknown='ignore',
        sparse_output=False
    )
    numeric_features = [
        col for col in data.columns if col not in ordinal_features + nominal_features
    ]
    scaler = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_encoder, ordinal_features),
            ('nominal', nominal_encoder, nominal_features),
            ('numeric', scaler, numeric_features)
        ] 
    )

    X = preprocessor.fit_transform(data)

    feature_names = preprocessor.get_feature_names_out()

    encoded_df = pd.DataFrame(
        X,
        columns=feature_names,
        index=data.index
    )

    return encoded_df


# FULL PIPELINE

def preprocess_pipeline(input_path, output_path):

    data = load_data(input_path)

    data = handle_structural_missing(data)
    data = handle_accidental_missing(data)

    encoded_data = encode_data(data)

    encoded_data.to_csv(output_path, index=False)

    return encoded_data
   
    