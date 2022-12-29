import numpy as np
import pandas as pd

NAMES_MAP = {
    'v2a1': 'monthly_rent',     # REVIEW nan => 0
    'hacdor': 'is_overcrowd_by_bedrooms',
    'rooms': 'number_rooms',
    'hacapo': 'is_overcrowd_by_rooms',
    'v14a': 'has_toilet',
    'refrig': 'has_refrigerator',
    'v18q': 'has_tablet',
    'v18q1': 'number_tablet',   # REVIEW nan, related with has_tablet => 0
    'r4h1': 'male_12-',
    'r4h2': 'male_12+',
    'r4h3': 'total_males',
    'r4m1': 'female_12-',
    'r4m2': 'female_12+',
    'r4m3': 'total_female',
    'r4t1': 'member_12-',
    'r4t2': 'member_12+',
    'r4t3': 'total_member',
    'tamhog': 'household_size',
    'tamviv': 'tenenment_size',
    'escolari': 'scholarship_years',
    'rez_esc': 'behind_school_years',   # REVIEW nan, may be related with scholarship_years => 0
    'paredblolad': 'wall_block_or_brick',
    'paredzocalo': 'wall_socket',
    'paredpreb': 'wall_prefabricated_or_cement',
    'pareddes': 'wall_waste_material',
    'paredmad': 'wall_wood',
    'paredzinc': 'wall_zink',
    'paredfibras': 'wall_natural_fibers',
    'paredother': 'wall_other',
    'pisomoscer': 'floor_mosaic_or_ceramic',
    'pisocemento': 'floor_cement',
    'pisoother': 'floor_other',
    'pisonatur': 'floor_natural_material',
    'pisonotiene': 'floor_no',
    'pisomadera': 'floor_wood',
    'techozinc': 'roof_metal_zink',
    'techoentrepiso': 'roof_fiber_cement_or_mezzanine',
    'techocane': 'roof_natural_fibers',
    'techootro': 'roof_other',
    'cielorazo': 'has_ceiling',
    'abastaguadentro': 'water_inside_dwelling',
    'abastaguafuera': 'water_outside_dwelling',
    'abastaguano': 'water_no',
    'public': 'electricity_public',
    'planpri': 'electricity_private',
    'noelec': 'electricity_no',
    'coopele': 'electricity_cooperative',
    'sanitario1': 'toilet_no',
    'sanitario2': 'toilet_sewer_or_cesspool',
    'sanitario3': 'toilet_septic_tank',
    'sanitario5': 'toilet_black_hole_or_letrine',
    'sanitario6': 'toilet_other',
    'energcocinar1': 'cook_energy_no',
    'energcocinar2': 'cook_energy_electricity',
    'energcocinar3': 'cook_energy_gas',
    'energcocinar4': 'cook_energy_wood_charcoal',
    'elimbasu1': 'rubbish_disposal_tanker_truck',
    'elimbasu2': 'rubbish_disposal_botan_hollow_or_buried',
    'elimbasu3': 'rubbish_disposal_burning',
    'elimbasu4': 'rubbish_disposal_throw_unoccupied_space',
    'elimbasu5': 'rubbish_disposal_throw_river_creek_sea',
    'elimbasu6': 'rubbish_disposal_other',
    'epared1': 'state_wall_bad',
    'epared2': 'state_wall_regular',
    'epared3': 'state_wall_good',
    'etecho1': 'state_roof_bad',
    'etecho2': 'state_roof_regular',
    'etecho3': 'state_roof_good',
    'eviv1': 'state_floor_bad',
    'eviv2': 'state_floor_regular',
    'eviv3': 'state_floor_good',
    'dis': 'is_disable',
    'male': 'is_male',
    'female': 'is_female',
    'estadocivil1': 'civil_state_10-',
    'estadocivil2': 'civil_state_free_or_coupled_union',
    'estadocivil3': 'civil_state_married',
    'estadocivil4': 'civil_state_divorced',
    'estadocivil5': 'civil_state_separated',
    'estadocivil6': 'civil_state_widow',
    'estadocivil7': 'civil_state_single',
    'parentesco1': 'kinship_household_head',
    'parentesco2': 'kinship_partner',
    'parentesco3': 'kinship_children',
    'parentesco4': 'kinship_stepchildren',
    'parentesco5': 'kinship_children_in_low',
    'parentesco6': 'kinship_grandchildren',
    'parentesco7': 'kinship_parent',
    'parentesco8': 'kinship_parent_in_low',
    'parentesco9': 'kinship_sibling',
    'parentesco10': 'kinship_sibling_in_low',
    'parentesco11': 'kinship_other_family',
    'parentesco12': 'kinship_not_family',
    'hogar_nin': 'children_19-',
    'hogar_adul': 'adults_number',
    'hogar_mayor': 'member_65+',
    'dependency': 'dependency_rate',
    'edjefe': 'male_head_education_years',
    'edjefa': 'female_head_education_years',
    'meaneduc': 'education_years_mean_18+',     # REVIEW nan => mean(scholarship_years)
    'instlevel1': 'education_no',
    'instlevel2': 'education_primary_incomplete',
    'instlevel3': 'education_primary_complete',
    'instlevel4': 'education_secondary_incomplete',
    'instlevel5': 'education_secondary_complete',
    'instlevel6': 'education_secondary_technical_incomplete',
    'instlevel7': 'education_secondary_technical_complete',
    'instlevel8': 'education_undergraduate_and_higher',
    'instlevel9': 'education_postgraduate',
    'bedrooms': 'bedrooms_number',
    'overcrowding': 'members_per_room',
    'tipovivi1': 'dwelling_type_own_and_paid',
    'tipovivi2': 'dwelling_type_own_and_paying',
    'tipovivi3': 'dwelling_type_rented',
    'tipovivi4': 'dwelling_type_precarious',
    'tipovivi5': 'dwelling_type_other',
    'computer': 'has_computer',
    'television': 'has_television',
    'mobilephone': 'has_mobile_phone',
    'qmobilephone': 'mobile_phone_number',
    'lugar1': 'region_central',
    'lugar2': 'region_chorotega',
    'lugar3': 'region_pacifico_central',
    'lugar4': 'region_brunca',
    'lugar5': 'region_huetar_atlantica',
    'lugar6': 'region_huetar_norte',
    'area1': 'urbana',
    'area2': 'rural',
    'sqbescolari': 'scholarship_years_sqd',
    'sqbage': 'age_sqd',
    'sqbhogar_total': 'total_household_sqd',
    'sqbedjefe': 'head_education_sqd',
    'sqbhogar_nin': 'children_19-_sqd',
    'sqbovercrowding': 'members_per_room_sqd',
    'sqbdependency': 'dependency_rate_sqd',
    'sqbmeaned': 'education_years_mean_18+_sqd',    # REVIEW nan, related with education_years_mean_18+ => education_years_mean_18+^2
}

KEEP_FEATURES = ['idhogar', 'age']

INDEX_KEY = 'idhogar'

ASSIGN_MAP = {
    # impute NaN 
    'monthly_rent': lambda df: np.where(
        df['monthly_rent'].isna(),
        0,
        df['monthly_rent']
    ),
    'number_tablet': lambda df: np.where(
        df['number_tablet'].isna(),
        0,
        df['number_tablet']
    ),
    'behind_school_years': lambda df: np.where(
        df['behind_school_years'].isna(),
        0,
        df['behind_school_years']
    ),
    'education_years_mean_18+': lambda df: np.where(
        df['education_years_mean_18+'].isna(),
        df.groupby(INDEX_KEY)['scholarship_years'].transform('mean'),
        df['education_years_mean_18+']
    ),
    'education_years_mean_18+_sqd': lambda df: np.where(
        df['education_years_mean_18+_sqd'].isna(),
        df['education_years_mean_18+'] ** 2,
        df['education_years_mean_18+_sqd']
    ),

    # change type of education years
    'male_head_education_years': lambda df: 
        df['male_head_education_years'] \
            .replace({'yes': 1, 'no': 0}) \
            .astype(int),
    'female_head_education_years': lambda df: 
        df['female_head_education_years'] \
            .replace({'yes': 1, 'no': 0}) \
            .astype(int),
    'dependency_rate': lambda df: 
        df['dependency_rate'] \
            .replace({'yes': 1, 'no': 0}) \
            .astype(float)
}


AGGREGATE_MAP = {
    'sum': [
        'is_disable', 'kinship_household_head', 'kinship_partner',
        'kinship_children', 'kinship_stepchildren', 'kinship_children_in_low',
        'kinship_grandchildren', 'kinship_parent', 'kinship_parent_in_low',
        'kinship_sibling', 'kinship_sibling_in_low', 'kinship_other_family',
        'kinship_not_family'
    ],
    'mean': [
        # proportions of specific population in a household
        'is_disable', 'is_male', 'is_female', 'scholarship_years_sqd',
        'age_sqd', 'children_19-_sqd', 'members_per_room_sqd', 'children_19-',
        'adults_number', 'member_65+', 'age'
    ]
}
